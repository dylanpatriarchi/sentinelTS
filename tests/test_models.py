from __future__ import annotations

import numpy as np
import pytest

from sentinelts.models import AnomalyDetector, IsolationForestDetector


def test_isolation_forest_scores_outliers_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(500, 4))
    det = IsolationForestDetector(n_estimators=100, seed=0).fit(normal)
    inliers = det.score(rng.normal(size=(50, 4)))
    outliers = det.score(rng.normal(loc=6.0, size=(50, 4)))
    assert outliers.min() > np.median(inliers)


def test_isolation_forest_is_deterministic_with_seed():
    X = np.random.default_rng(1).normal(size=(200, 3))
    a = IsolationForestDetector(seed=3).fit(X).score(X)
    b = IsolationForestDetector(seed=3).fit(X).score(X)
    assert np.array_equal(a, b)


def test_detector_protocol_and_unfitted_guard():
    det = IsolationForestDetector()
    assert isinstance(det, AnomalyDetector)
    with pytest.raises(RuntimeError):
        det.score(np.zeros((1, 2)))
