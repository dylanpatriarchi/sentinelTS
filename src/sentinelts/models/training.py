"""Reusable training loop for reconstruction-based detectors (LSTM-AE, Transformer-AE).

Early stopping: after each epoch we measure the reconstruction loss on held-out
*healthy* windows from engines not used for gradient updates. When it stops
improving for ``patience`` epochs we stop and restore the best weights. This
prevents the model from memorising the training engines' noise.
"""

from __future__ import annotations

import copy
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


@torch.no_grad()
def reconstruction_errors(
    model: nn.Module, X: np.ndarray, batch_size: int = 512, device: str = "cpu"
) -> np.ndarray:
    """Squared error per sample, time step and feature: shape (n, window, n_features)."""
    model.eval()
    out = []
    for (xb,) in _loader(X, batch_size, shuffle=False):
        xb = xb.to(device)
        out.append(((model(xb) - xb) ** 2).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, *X.shape[1:]), dtype=np.float32)


def train_reconstruction(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: TrainingConfig,
    seed: int = 0,
) -> TrainingHistory:
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    history = TrainingHistory()
    best_state, best_val, stale = None, float("inf"), 0

    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for (xb,) in _loader(X_train, cfg.batch_size, shuffle=True, seed=seed + epoch):
            xb = xb.to(cfg.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        history.train_loss.append(total / n)

        val = float(reconstruction_errors(model, X_val, device=cfg.device).mean())
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
