"""Causal sliding windows, one per row.

For a row at cycle t of engine u, its window holds the feature vectors of
cycles t-W+1 .. t of the *same* engine (never the future, never another
engine). The first W-1 cycles of an engine lack history; we left-pad by
repeating the engine's first observation. Training and scoring use the same
padding rule, so the model sees padded windows as part of "normal".

Returning exactly one window per row keeps scores aligned with the DataFrame,
whatever the window length (W=1 reproduces point-wise models).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def make_windows(df: pd.DataFrame, columns: list[str], window: int) -> np.ndarray:
    """Return a float32 array of shape (len(df), window, len(columns))."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if not df["unit"].is_monotonic_increasing:
        raise ValueError("df must be sorted by unit, cycle")

    values = df[columns].to_numpy(dtype=np.float32)
    out = np.empty((len(df), window, len(columns)), dtype=np.float32)
    # Positions where each unit starts/ends inside df (df is sorted by unit).
    bounds = np.flatnonzero(np.diff(df["unit"].to_numpy(), prepend=np.nan, append=np.nan))
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        series = values[start:end]
        padded = np.concatenate([np.repeat(series[:1], window - 1, axis=0), series])
        # sliding_window_view -> (n, F, W); move time axis before features.
        out[start:end] = sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)
    return out
