"""Common interface for every detector (Isolation Forest, LSTM-AE, Transformer...).

Input: ``X`` of shape (n_samples, window, n_features), built by
``features.make_windows`` -- one causal window per scored cycle. Point-wise
models declare ``window = 1``.

``groups`` holds the engine id of each sample. Detectors that need an internal
hold-out (e.g. early stopping) must split by group, never by sample, for the
same reason the outer protocol splits by engine.

Convention: ``score`` returns one value per sample, **higher = more anomalous**.
Detectors are trained on healthy data only and never see anomaly labels.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyDetector(Protocol):
    window: int

    def fit(self, X: np.ndarray, groups: np.ndarray) -> AnomalyDetector: ...

    def score(self, X: np.ndarray) -> np.ndarray: ...
