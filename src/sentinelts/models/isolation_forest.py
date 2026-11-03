"""Isolation Forest baseline.

Idea (Liu, Ting & Zhou, 2008): build many random binary trees, each split picks a
random feature and a random cut between its min and max. Anomalies are "few and
different", so they get isolated in fewer splits -> shorter average path length.
The anomaly score is a normalised function of that average depth.

It is a point-wise model: it sees one feature vector per cycle and knows nothing
about time, except what we inject through rolling features.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | str = "auto",
        max_features: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            # contamination only affects sklearn's built-in predict(); we never
            # use it -- the threshold comes from validation data instead.
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model.fit(np.asarray(X, dtype=float))
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        # sklearn's score_samples is HIGHER for normal points; flip the sign.
        return -self.model.score_samples(np.asarray(X, dtype=float))
