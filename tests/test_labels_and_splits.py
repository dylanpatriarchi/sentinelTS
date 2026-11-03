from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelts.data.labels import AMBIGUOUS, ANOMALOUS, HEALTHY, add_anomaly_labels
from sentinelts.data.splits import UnitSplit, healthy_prefix, select_units, split_units


def test_label_boundaries_are_inclusive_on_the_anomalous_side():
    df = pd.DataFrame({"rul": [101, 100, 31, 30, 0]})
    labels = add_anomaly_labels(df, anomaly_rul=30, healthy_rul=100)["label"].tolist()
    assert labels == [HEALTHY, AMBIGUOUS, AMBIGUOUS, ANOMALOUS, ANOMALOUS]


def test_labels_reject_inverted_thresholds():
    with pytest.raises(ValueError):
        add_anomaly_labels(pd.DataFrame({"rul": [1]}), anomaly_rul=50, healthy_rul=10)


def _fleet(n_units: int = 10, length: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(1, n_units + 1), length),
            "cycle": np.tile(np.arange(1, length + 1), n_units),
        }
    )


def test_split_is_disjoint_complete_and_reproducible():
    df = _fleet(10)
    a = split_units(df, val_fraction=0.2, seed=7)
    b = split_units(df, val_fraction=0.2, seed=7)
    assert np.array_equal(a.fit_units, b.fit_units)
    assert len(a.val_units) == 2
    assert set(a.fit_units) | set(a.val_units) == set(range(1, 11))
    assert not set(a.fit_units) & set(a.val_units)


def test_rows_of_one_engine_never_cross_the_split():
    df = _fleet(10)
    s = split_units(df, seed=0)
    fit, val = select_units(df, s.fit_units), select_units(df, s.val_units)
    assert len(fit) + len(val) == len(df)
    assert not set(fit.unit) & set(val.unit)


def test_unit_split_guards_against_overlap():
    with pytest.raises(ValueError, match="both"):
        UnitSplit(fit_units=np.array([1, 2]), val_units=np.array([2, 3]))


def test_healthy_prefix_selects_by_cycle_not_rul():
    df = _fleet(3, length=5)
    out = healthy_prefix(df, n_cycles=2)
    assert out["cycle"].max() == 2
    assert len(out) == 6
