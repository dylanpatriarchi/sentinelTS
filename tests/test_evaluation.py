from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sentinelts.evaluation import event_metrics, point_metrics, quantile_threshold


def _scored() -> pd.DataFrame:
    """Two engines. Unit 1 degrades and is caught; unit 2 has a false alarm and is missed."""
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "rul": [150, 60, 20, 0, 150, 120, 10, 0],
            "label": [0, -1, 1, 1, 0, 0, 1, 1],
            "score": [0.1, 0.6, 0.9, 0.95, 0.2, 0.7, 0.3, 0.4],
        }
    )


def test_point_metrics_hand_computed():
    m = point_metrics(_scored(), threshold=0.5)
    # ambiguous row dropped -> 7 points; preds>0.5 on: u1(0.9,0.95) u2(0.7)
    assert m.n_points == 7
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 4)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m.anomaly_ratio == pytest.approx(4 / 7)
    # early warning: ambiguous {0.6} vs healthy {0.1, 0.2, 0.7} -> 2 of 3 pairs ranked right
    assert m.early_warning_auc == pytest.approx(2 / 3)


def test_perfect_detector_scores_one():
    df = _scored().assign(score=lambda d: (d["label"] == 1).astype(float))
    m = point_metrics(df, threshold=0.5)
    assert m.roc_auc == 1.0 and m.pr_auc == 1.0 and m.f1 == 1.0


def test_event_metrics_hand_computed():
    m = event_metrics(_scored(), threshold=0.5)
    assert m.n_units_with_anomaly == 2
    assert m.detection_rate == 0.5
    assert m.median_lead_time == 60  # first non-healthy alarm of unit 1 is in the ambiguous zone
    assert m.false_alarm_unit_rate == 0.5
    assert m.false_alarm_point_rate == pytest.approx(1 / 3)


def test_truncated_engines_count_only_for_false_alarms():
    df = pd.DataFrame({"unit": [1, 1], "rul": [200, 150], "label": [0, 0], "score": [0.0, 0.9]})
    m = event_metrics(df, threshold=0.5)
    assert m.n_units_with_anomaly == 0
    assert math.isnan(m.detection_rate)
    assert m.false_alarm_unit_rate == 1.0


def test_nan_scores_rejected():
    df = _scored()
    df.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        point_metrics(df, threshold=0.5)


def test_quantile_threshold_sets_false_alarm_rate():
    healthy = np.arange(1000, dtype=float)
    thr = quantile_threshold(healthy, q=0.99)
    assert np.mean(healthy > thr) == pytest.approx(0.01, abs=1e-3)
    with pytest.raises(ValueError):
        quantile_threshold(np.array([]))
