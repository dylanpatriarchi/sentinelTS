"""Common interface for every detector (Isolation Forest, LSTM-AE, Transformer...).

Convention: ``score`` returns one value per input row, **higher = more anomalous**.
Detectors are trained on healthy data only (semi-supervised / one-class setting):
they never see an anomaly label.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyDetector(Protocol):
    def fit(self, X: np.ndarray) -> AnomalyDetector: ...

    def score(self, X: np.ndarray) -> np.ndarray: ...
