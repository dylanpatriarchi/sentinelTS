"""CLI and run-record tests, end to end on the synthetic fleet."""

from __future__ import annotations

import json

import pytest

from sentinelts.cli import build_parser, main
from sentinelts.models.registry import MODEL_DEFAULTS, build_factory, model_params
from sentinelts.runs import config_hash, load_runs, summarize


def test_parser_exposes_registry_defaults():
    args = build_parser().parse_args(["run", "--model", "lstm_ae"])
    assert args.window == MODEL_DEFAULTS["lstm_ae"]["window"]
    assert args.lr == pytest.approx(1e-3)
    assert args.evaluate_test is False


def test_model_params_only_keeps_relevant_keys():
    p = model_params("iforest", {"n_estimators": 10, "window": 99, "lr": 1.0})
    assert p == {"n_estimators": 10}
    assert "lr" in model_params("transformer", {})
    with pytest.raises(ValueError):
        model_params("svm", {})


def test_factories_build_detectors_with_declared_window():
    for model in MODEL_DEFAULTS:
        det = build_factory(model, {"window": 7}, seed=0)()
        assert det.window == (1 if model == "iforest" else 7)


def test_config_hash_ignores_seed_but_not_settings():
    exp = {"subset": "FD001", "seed": 1, "rolling_window": 0}
    same = config_hash("iforest", {"n_estimators": 5}, exp | {"seed": 2})
    assert config_hash("iforest", {"n_estimators": 5}, exp) == same
    assert config_hash("iforest", {"n_estimators": 6}, exp) != same


def test_run_then_compare_end_to_end(synthetic_root, tmp_path, capsys):
    art = tmp_path / "artifacts"
    for seed in (0, 1):
        code = main(
            [
                "run",
                "--model",
                "iforest",
                "--n-estimators",
                "20",
                "--seed",
                str(seed),
                "--root",
                str(synthetic_root),
                "--artifacts",
                str(art),
            ]
        )
        assert code == 0
    out = capsys.readouterr().out
    assert "[validation] early_warning_auc=" in out and "[test]" not in out

    records = list(art.glob("iforest/FD001/*/seed*/metrics.json"))
    assert len(records) == 2
    rec = json.loads(records[0].read_text())
    assert rec["params"] == {"n_estimators": 20}
    assert "test" not in rec

    table = summarize(load_runs(art), ["early_warning_auc"])
    assert len(table) == 1 and table["n_seeds"].iloc[0] == 2

    assert main(["compare", "--artifacts", str(art)]) == 0
    assert "early_warning_auc_mean" in capsys.readouterr().out


def test_compare_on_empty_dir_fails_cleanly(tmp_path):
    assert main(["compare", "--artifacts", str(tmp_path)]) == 1


def test_explore_runs_on_synthetic_data(synthetic_root, capsys):
    assert main(["explore", "--root", str(synthetic_root), "--subset", "FD001"]) == 0
    assert "train_engines" in capsys.readouterr().out
