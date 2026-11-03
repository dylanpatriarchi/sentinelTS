"""Choosing an alarm threshold without looking at the evaluation set.

An anomaly detector outputs a continuous score; turning it into alarms needs a
threshold. Picking the threshold that maximises F1 *on the test set* is a classic
leakage bug (it reports an upper bound, not a real performance). Here the
threshold is a high quantile of scores on *healthy validation data only*, so
it is equivalent to fixing a target false-alarm rate (e.g. q=0.99 -> ~1%).
"""

from __future__ import annotations

import numpy as np


def quantile_threshold(healthy_scores: np.ndarray, q: float = 0.99) -> float:
    scores = np.asarray(healthy_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Need at least one healthy score to set a threshold")
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or inf")
    return float(np.quantile(scores, q))
