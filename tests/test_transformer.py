from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinelts.models import AnomalyDetector, TrainingConfig, TransformerDetector
from sentinelts.models.training import forecast_error
from sentinelts.models.transformer import TransformerForecaster

FAST = TrainingConfig(epochs=40, batch_size=64, lr=3e-3, patience=8)


def _healthy(n: int, window: int = 12, n_features: int = 3, seed: int = 0) -> np.ndarray:
    """Noisy sinusoids: the next value is predictable from the past."""
    rng = np.random.default_rng(seed)
    t = np.arange(window)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, n_features))
    noise = 0.05 * rng.normal(size=(n, window, n_features))
    return (np.sin(0.5 * t + phase) + noise).astype(np.float32)


def test_forward_predicts_one_cycle():
    model = TransformerForecaster(n_features=5, context=9, d_model=16, nhead=2)
    assert model(torch.randn(4, 9, 5)).shape == (4, 5)
    with pytest.raises(ValueError, match="context"):
        model(torch.randn(4, 8, 5))


def test_forecast_never_uses_the_target_cycle():
    """Changing x_t must not change the prediction of x_t (no copying possible)."""
    torch.manual_seed(0)
    model = TransformerForecaster(n_features=3, context=11, d_model=16, nhead=2).eval()
    x = torch.randn(2, 12, 3)
    x2 = x.clone()
    x2[:, -1] += 100.0
    with torch.no_grad():
        assert torch.equal(model(x[:, :-1]), model(x2[:, :-1]))
        assert (forecast_error(model, x2) > forecast_error(model, x)).all()


def test_detector_learns_and_flags_unpredictable_last_cycle():
    X = _healthy(400)
    groups = np.repeat(np.arange(20), 20)
    det = TransformerDetector(window=12, d_model=16, nhead=2, training=FAST).fit(X, groups)
    hist = det.history
    assert hist["val_loss"][hist["best_epoch"]] < 0.5 * hist["val_loss"][0]
    normal = det.score(_healthy(50, seed=2))
    jumped = _healthy(50, seed=3)
    jumped[:, -1, :] += 2.0  # sudden jump at the current cycle
    assert np.median(det.score(jumped)) > np.percentile(normal, 95)


def test_protocol_validation_and_determinism():
    det = TransformerDetector(window=12)
    assert isinstance(det, AnomalyDetector)
    with pytest.raises(RuntimeError):
        det.score(_healthy(2))
    with pytest.raises(ValueError):
        TransformerDetector(window=1)
    X, groups = _healthy(100), np.repeat(np.arange(5), 20)
    cfg = TrainingConfig(epochs=2)
    a = TransformerDetector(window=12, d_model=16, nhead=2, training=cfg, seed=1).fit(X, groups)
    b = TransformerDetector(window=12, d_model=16, nhead=2, training=cfg, seed=1).fit(X, groups)
    assert np.allclose(a.score(X[:10]), b.score(X[:10]))
