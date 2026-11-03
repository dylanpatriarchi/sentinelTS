"""Reusable training loop for neural detectors (LSTM-AE, Transformer forecaster).

The objective is pluggable through an ``ErrorFn``: reconstruction (autoencoders)
or one-step-ahead forecasting. The same function gives the training loss and,
at inference, the anomaly score -- so "what the model was trained to do well on
healthy data" and "what we measure as abnormal" are guaranteed to match.

Early stopping: after each epoch we measure the reconstruction loss on held-out
*healthy* windows from engines not used for gradient updates. When it stops
improving for ``patience`` epochs we stop and restore the best weights. This
prevents the model from memorising the training engines' noise.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    grad_clip: float = 1.0  # LSTMs can have exploding gradients
    device: str = "cpu"


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
        }


def set_seed(seed: int) -> None:
    """Seed torch (weight init, dropout). NumPy code in this repo uses explicit Generators."""
    torch.manual_seed(seed)


def _loader(X: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


ErrorFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]
"""Maps (model, batch of windows) to squared errors with the feature axis last."""


def reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Autoencoders: reconstruct the whole window. Shape (B, W, F)."""
    return (model(x) - x) ** 2


def forecast_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forecasters: predict the last cycle from the previous ones. Shape (B, 1, F)."""
    return (model(x[:, :-1]) - x[:, -1]).unsqueeze(1) ** 2


def engine_holdout_mask(groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask selecting all samples of a random subset of engines (for early stopping)."""
    units = np.unique(groups)
    if len(units) < 2:
        raise ValueError("Need at least 2 engines to hold one out for early stopping")
    rng = np.random.default_rng(seed)
    n_out = max(1, round(len(units) * fraction))
    return np.isin(groups, rng.choice(units, size=n_out, replace=False))


@torch.no_grad()
def compute_errors(
    model: nn.Module,
    X: np.ndarray,
    error_fn: ErrorFn = reconstruction_error,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Squared errors for every sample, as returned by ``error_fn`` (feature axis last)."""
    model.eval()
    out = [error_fn(model, xb.to(device)).cpu().numpy() for (xb,) in _loader(X, batch_size, False)]
    if not out:
        raise ValueError("X is empty")
    return np.concatenate(out)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: TrainingConfig,
    error_fn: ErrorFn = reconstruction_error,
    seed: int = 0,
) -> TrainingHistory:
    """Minimise mean ``error_fn`` on X_train; early-stop on mean ``error_fn`` on X_val."""
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = TrainingHistory()
    best_state, best_val, stale = None, float("inf"), 0

    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for (xb,) in _loader(X_train, cfg.batch_size, shuffle=True, seed=seed + epoch):
            xb = xb.to(cfg.device)
            opt.zero_grad()
            loss = error_fn(model, xb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        history.train_loss.append(total / n)

        val = float(compute_errors(model, X_val, error_fn, device=cfg.device).mean())
        history.val_loss.append(val)
        if val < best_val - 1e-6:
            best_val, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
            history.best_epoch = epoch
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
