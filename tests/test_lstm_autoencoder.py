from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinelts.models import (
    AnomalyDetector,
    LSTMAutoencoder,
    LSTMAutoencoderDetector,
    TrainingConfig,
)
from sentinelts.models.training import TrainingConfig as TC
from sentinelts.models.training import train_reconstruction

FAST = TrainingConfig(epochs=40, batch_size=64, lr=5e-3, patience=8)


def _healthy(n: int, window: int = 12, n_features: int = 3, seed: int = 0) -> np.ndarray:
    """Noisy sinusoids with random phase: a learnable 'healthy' dynamic."""
    rng = np.random.default_rng(seed)
    t = np.arange(window)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, n_features))
    return (np.sin(0.5 * t + phase) + 0.05 * rng.normal(size=(n, window, n_features))).astype(
        np.float32
    )


def test_forward_preserves_shape():
    model = LSTMAutoencoder(n_features=5, hidden_size=16, latent_size=4)
    x = torch.randn(7, 20, 5)
    assert model(x).shape == x.shape


def test_training_reduces_loss():
    torch.manual_seed(0)
    # 3 independent phases need ~6 latent dims (sin/cos each): give it 8.
    model = LSTMAutoencoder(n_features=3, hidden_size=16, latent_size=8)
    hist = train_reconstruction(model, _healthy(256), _healthy(64, seed=1), FAST)
    assert hist.val_loss[hist.best_epoch] < 0.5 * hist.val_loss[0]


def test_early_stopping_stops_and_restores_best():
    model = LSTMAutoencoder(n_features=3, hidden_size=8, latent_size=2)
    cfg = TC(epochs=200, batch_size=64, lr=5e-3, patience=3)
    hist = train_reconstruction(model, _healthy(128), _healthy(32, seed=1), cfg)
    assert len(hist.val_loss) < 200
    assert hist.best_epoch == int(np.argmin(hist.val_loss))


def test_detector_scores_anomalous_windows_higher():
    X = _healthy(400)
    groups = np.repeat(np.arange(20), 20)
    det = LSTMAutoencoderDetector(window=12, hidden_size=16, latent_size=4, training=FAST)
    det.fit(X, groups)
    normal = det.score(_healthy(50, seed=2))
    drifted = _healthy(50, seed=3)
    drifted[:, 6:, :] += 2.0  # level shift in the second half of the window
    assert np.median(det.score(drifted)) > np.percentile(normal, 95)


def test_detector_is_deterministic_with_seed():
    X, groups = _healthy(200), np.repeat(np.arange(10), 20)
    cfg = TrainingConfig(epochs=3, batch_size=64)
    a = LSTMAutoencoderDetector(window=12, hidden_size=8, training=cfg, seed=1).fit(X, groups)
    b = LSTMAutoencoderDetector(window=12, hidden_size=8, training=cfg, seed=1).fit(X, groups)
    assert np.allclose(a.score(X[:20]), b.score(X[:20]))


def test_early_stopping_holds_out_whole_engines():
    det = LSTMAutoencoderDetector(es_fraction=0.2, seed=0)
    groups = np.repeat(np.arange(10), 5)
    mask = det._split_by_engine(groups)
    assert mask.sum() == 10  # 2 engines x 5 samples
    assert not set(groups[mask]) & set(groups[~mask])


def test_protocol_and_input_validation():
    det = LSTMAutoencoderDetector(window=12)
    assert isinstance(det, AnomalyDetector)
    with pytest.raises(RuntimeError):
        det.score(_healthy(2))
    with pytest.raises(ValueError, match="shape"):
        det.fit(_healthy(10, window=5), np.arange(10))
    with pytest.raises(ValueError):
        LSTMAutoencoderDetector(score_on="mean")
