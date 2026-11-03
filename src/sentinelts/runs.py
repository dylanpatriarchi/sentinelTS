"""Persisting experiment runs and comparing them across seeds.

Layout:  <root>/<model>/<subset>/<label>-<config_hash>/seed<seed>/
             metrics.json      config, metrics, git revision, training history
             val_scores.csv    per-cycle scores/alarms on validation engines
             test_scores.csv   only if the test set was evaluated

``config_hash`` covers every setting except the seed, so runs that differ only
by seed (= engine split) are grouped together by ``summarize``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelts.experiment import ExperimentResult

SPLITS = ("validation", "test")


def config_hash(model: str, params: dict[str, Any], experiment: dict[str, Any]) -> str:
    payload = {"model": model, "params": params, "experiment": dict(experiment)}
    payload["experiment"].pop("seed", None)
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def run_label(model: str, params: dict[str, Any], experiment: dict[str, Any]) -> str:
    """Short human-readable name; uniqueness comes from the hash, not from this."""
    parts = [f"w{experiment['rolling_window']}", f"k{experiment['score_smoothing']}"]
    if "window" in params:
        parts.insert(0, f"W{params['window']}")
    if model == "lstm_ae":
        parts.append(params["score_on"])
    return "_".join(parts)


def git_revision(cwd: Path | None = None) -> dict[str, Any] | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(dirty)}


def save_run(result: ExperimentResult, model: str, params: dict[str, Any], root: Path) -> Path:
    exp = result.config
    h = config_hash(model, params, exp)
    label = run_label(model, params, exp)
    run_dir = root / model / exp["subset"] / f"{label}-{h}" / f"seed{exp['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "model": model,
        "label": label,
        "config_hash": h,
        "params": params,
        "git": git_revision(),
    } | result.summary()
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
    result.validation.scored.to_csv(run_dir / "val_scores.csv", index=False)
    if result.test is not None:
        result.test.scored.to_csv(run_dir / "test_scores.csv", index=False)
    return run_dir


def load_runs(root: Path) -> pd.DataFrame:
    """One row per (run, split) with flattened point and event metrics."""
    rows = []
    for path in sorted(Path(root).glob("*/*/*/seed*/metrics.json")):
        rec = json.loads(path.read_text())
        for split in SPLITS:
            if split not in rec:
                continue
            row = {
                "model": rec["model"],
                "subset": rec["config"]["subset"],
                "label": rec["label"],
                "config_hash": rec["config_hash"],
                "seed": rec["config"]["seed"],
                "split": split,
            }
            row |= rec[split]["point"] | rec[split]["event"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, metrics: list[str], split: str = "validation") -> pd.DataFrame:
    """Mean and std of ``metrics`` over seeds, per configuration."""
    if runs.empty:
        return runs
    df = runs[runs["split"] == split]
    keys = ["subset", "model", "label", "config_hash"]
    grouped = df.groupby(keys)
    out = grouped[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out["n_seeds"] = grouped.size()
    return out.reset_index().sort_values(["subset", f"{metrics[0]}_mean"], ascending=[True, False])
