"""Leakage-safe splitting.

Rules enforced here:
1. Split by *engine unit*, never by row. Rows of the same engine are strongly
   autocorrelated; a row-level random split puts near-duplicate neighbours in
   train and validation and inflates every metric.
2. The official C-MAPSS ``test`` files are the final hold-out. They are never
   passed to this function: they are opened once, at the very end.
3. Model fitting only sees the *early-life* cycles of the ``fit`` units
   (``healthy_prefix``), selected by cycle index -- information available in
   production -- not by RUL, which requires knowing the future failure time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UnitSplit:
    fit_units: np.ndarray
    val_units: np.ndarray

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.fit_units, self.val_units)
        if overlap.size:
            raise ValueError(f"Units in both fit and val: {overlap.tolist()}")


def split_units(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> UnitSplit:
    """Randomly assign whole engines to ``fit`` or ``val`` (reproducible via ``seed``)."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, round(len(units) * val_fraction))
    return UnitSplit(
        fit_units=np.sort(shuffled[n_val:]),
        val_units=np.sort(shuffled[:n_val]),
    )


def select_units(df: pd.DataFrame, units: np.ndarray) -> pd.DataFrame:
    return df[df["unit"].isin(units)].reset_index(drop=True)


def healthy_prefix(df: pd.DataFrame, n_cycles: int) -> pd.DataFrame:
    """First ``n_cycles`` of each unit: the data a detector is allowed to learn 'normal' from."""
    if n_cycles <= 0:
        raise ValueError("n_cycles must be positive")
    return df[df["cycle"] <= n_cycles].reset_index(drop=True)
