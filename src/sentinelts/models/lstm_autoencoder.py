"""LSTM Autoencoder for multivariate time-series anomaly detection.

Idea (Malhotra et al., 2016): train a network to *reconstruct* windows of
healthy sensor data through a narrow bottleneck. It can only learn to compress
patterns it has seen, i.e. healthy dynamics. On degraded data the
reconstruction gets worse, and the reconstruction error is the anomaly score.

Architecture ("repeat-vector" encoder-decoder):

    x (B, W, F) ─► LSTM encoder ─► last hidden state h_W (B, H)
               ─► Linear ─► z (B, latent)            <- bottleneck
               ─► Linear ─► repeated W times (B, W, H)
               ─► LSTM decoder ─► Linear ─► x̂ (B, W, F)

The bottleneck (latent << W*F) is what stops the network from learning the
identity function, which would reconstruct anomalies just as well.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from sentinelts.models.training import (
    TrainingConfig,
    compute_errors,
    engine_holdout_mask,
    set_seed,
    train_model,
)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        latent_size: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            n_features, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout
        )
        self.output = nn.Linear(hidden_size, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)  # h: (num_layers, B, H)
        return self.to_latent(h[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        seq = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.decoder(seq)
        return self.output(dec)


ScoreOn = Literal["window", "last"]


class LSTMAutoencoderDetector:
    """Fit/score wrapper following ``models.base.AnomalyDetector``.

    score_on:
        "window" -> mean squared error over the whole window (smoother, lags);
        "last"   -> error of the most recent cycle only (reacts faster, noisier).
    es_fraction:
        Fraction of *training engines* held out for early stopping.
    """

    def __init__(
        self,
        window: int = 30,
        hidden_size: int = 64,
        latent_size: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        score_on: ScoreOn = "window",
        es_fraction: float = 0.15,
        training: TrainingConfig | None = None,
        seed: int = 0,
    ) -> None:
        if score_on not in ("window", "last"):
            raise ValueError("score_on must be 'window' or 'last'")
        self.window = window
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.score_on = score_on
        self.es_fraction = es_fraction
        self.training = training or TrainingConfig()
        self.seed = seed
        self.model: LSTMAutoencoder | None = None
        self.history: dict = {}

    def fit(self, X: np.ndarray, groups: np.ndarray) -> LSTMAutoencoderDetector:
        if X.ndim != 3 or X.shape[1] != self.window:
            raise ValueError(f"Expected X of shape (n, {self.window}, n_features), got {X.shape}")
        set_seed(self.seed)
        es_mask = engine_holdout_mask(np.asarray(groups), self.es_fraction, self.seed)
        self.model = LSTMAutoencoder(
            X.shape[2], self.hidden_size, self.latent_size, self.num_layers, self.dropout
        )
        hist = train_model(self.model, X[~es_mask], X[es_mask], self.training, seed=self.seed)
        self.history = hist.as_dict()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Detector must be fitted before scoring")
        err = compute_errors(self.model, X, device=self.training.device)
        if self.score_on == "last":
            return err[:, -1, :].mean(axis=1).astype(float)
        return err.mean(axis=(1, 2)).astype(float)
