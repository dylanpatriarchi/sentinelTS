"""End-to-end tests of the experiment pipeline on a synthetic degrading fleet."""

from __future__ import annotations

import numpy as np
import pytest

from sentinelts.experiment import ExperimentConfig, run_experiment
from sentinelts.models import IsolationForestDetector


def _factory():
    return IsolationForestDetector(n_estimators=50, seed=0)


def test_pipeline_detects_injected_degradation(synthetic_root):
    res = run_experiment(synthetic_root, _factory, ExperimentConfig(rolling_window=5))
    assert res.validation.point["pr_auc"] > res.validation.point["anomaly_ratio"]
    assert res.validation.point["roc_auc"] > 0.8
    assert res.test is None  # test set untouched unless explicitly requested


def test_test_set_only_scored_on_request(synthetic_root):
    res = run_experiment(synthetic_root, _factory, ExperimentConfig(), evaluate_test=True)
    assert res.test is not None
    assert res.test.scored["unit"].nunique() == 6


def test_validation_and_test_data_do_not_influence_fitting(synthetic_root):
    """Corrupting the test file must not change anything learned (scores on val)."""
    cfg = ExperimentConfig(rolling_window=0)
    before = run_experiment(synthetic_root, _factory, cfg)
    data = np.loadtxt(synthetic_root / "test_FD001.txt")
    data[:, 5:] += 1000.0
    np.savetxt(synthetic_root / "test_FD001.txt", data, fmt="%.5f")
    after = run_experiment(synthetic_root, _factory, cfg)
    assert np.array_equal(before.validation.scored["score"], after.validation.scored["score"])
    assert before.threshold == after.threshold


def test_val_units_are_disjoint_from_training_units(synthetic_root):
    res = run_experiment(synthetic_root, _factory, ExperimentConfig())
    val_units = set(res.validation.scored["unit"])
    assert 0 < len(val_units) < 20


def test_warmup_cycles_are_never_scored(synthetic_root):
    res = run_experiment(synthetic_root, _factory, ExperimentConfig(warmup_cycles=10))
    assert res.validation.scored["cycle"].min() == 11


def test_invalid_window_rejected():
    with pytest.raises(ValueError, match="warmup"):
        ExperimentConfig(rolling_window=20, warmup_cycles=10)


def test_pipeline_runs_with_windowed_lstm_autoencoder(synthetic_root):
    from sentinelts.models import LSTMAutoencoderDetector, TrainingConfig

    def factory():
        return LSTMAutoencoderDetector(
            window=10, hidden_size=16, latent_size=4, training=TrainingConfig(epochs=5)
        )

    res = run_experiment(synthetic_root, factory, ExperimentConfig(rolling_window=0))
    assert res.validation.scored["cycle"].min() == 11  # same evaluated rows as point models
    assert "training_history" in res.extra
    assert np.isfinite(res.validation.scored["score"]).all()


def test_score_smoothing_is_causal_and_per_engine():
    from sentinelts.experiment import smooth_scores

    units = np.array([1, 1, 1, 2, 2])
    out = smooth_scores(np.array([1.0, 3.0, 5.0, 10.0, 20.0]), units, k=2)
    assert out.tolist() == [1.0, 2.0, 4.0, 10.0, 15.0]
    assert smooth_scores(np.array([1.0, 2.0]), np.array([1, 1]), k=1).tolist() == [1.0, 2.0]
