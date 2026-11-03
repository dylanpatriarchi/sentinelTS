"""Transformer forecaster for anomaly detection.

Why forecasting and not a Transformer autoencoder?
    Self-attention lets every output position look directly at its own input
    position, so an un-bottlenecked Transformer autoencoder can learn the
    identity and reconstruct anomalies perfectly. Instead we ask: "given the
    previous W-1 cycles, what should cycle t look like?" The model never sees
    x_t when predicting it, so it cannot copy; it can only learn healthy
    dynamics. Score = squared prediction error at cycle t (causal by
    construction: uses cycles t-W+1 .. t).

Architecture:

    x[t-W+1 .. t-1] (B, W-1, F) ─► Linear(F→d) + learned positional embedding
                                ─► N x TransformerEncoderLayer (self-attention + FFN)
                                ─► representation of the last position ─► Linear(d→F) ─► x̂_t

No causal mask is needed: every input position is already in the past of t.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sentinelts.models.training import (
    TrainingConfig,
    compute_errors,
    engine_holdout_mask,
    forecast_error,
    set_seed,
    train_model,
)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        context: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context = context
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,  # pre-LayerNorm: more stable training for small data
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_features)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        """past: (B, context, F) -> prediction of the next cycle (B, F)."""
        if past.size(1) != self.context:
            raise ValueError(f"Expected context length {self.context}, got {past.size(1)}")
        h = self.encoder(self.input_proj(past) + self.pos_embedding)
        return self.head(h[:, -1])


class TransformerDetector:
    """Fit/score wrapper following ``models.base.AnomalyDetector``.

    ``window`` = context + 1: the last cycle of each window is the forecast target.
    """

    def __init__(
        self,
        window: int = 30,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        es_fraction: float = 0.15,
        training: TrainingConfig | None = None,
        seed: int = 0,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 (at least one past cycle)")
        self.window = window
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.es_fraction = es_fraction
        self.training = training or TrainingConfig()
        self.seed = seed
        self.model: TransformerForecaster | None = None
        self.history: dict = {}

    def fit(self, X: np.ndarray, groups: np.ndarray) -> TransformerDetector:
        if X.ndim != 3 or X.shape[1] != self.window:
            raise ValueError(f"Expected X of shape (n, {self.window}, n_features), got {X.shape}")
        set_seed(self.seed)
        es_mask = engine_holdout_mask(np.asarray(groups), self.es_fraction, self.seed)
        self.model = TransformerForecaster(
            X.shape[2],
            self.window - 1,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dim_feedforward,
            self.dropout,
        )
        hist = train_model(
            self.model, X[~es_mask], X[es_mask], self.training, forecast_error, self.seed
        )
        self.history = hist.as_dict()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Detector must be fitted before scoring")
        err = compute_errors(self.model, X, forecast_error, device=self.training.device)
        return err.mean(axis=(1, 2)).astype(float)
