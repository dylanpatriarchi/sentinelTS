from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelts.data.cmapss import OPERATIONAL_SETTINGS, SENSORS
from sentinelts.features import ConditionNormalizer, add_rolling_features


def _two_condition_frame(n: int = 400, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Two regimes far apart in op-settings; sensors depend strongly on the regime."""
    rng = np.random.default_rng(seed)
    cond = np.arange(n) % 2
    df = pd.DataFrame({"unit": 1, "cycle": np.arange(1, n + 1)})
    df[OPERATIONAL_SETTINGS] = np.c_[cond * 40.0, cond * 0.8, np.full(n, 100.0)]
    for i, s in enumerate(SENSORS):
        df[s] = 100.0 * cond + i + rng.normal(scale=1.0, size=n) + shift
    df["s_1"] = 5.0  # constant sensor
    return df


def test_per_condition_zscore_on_training_data():
    df = _two_condition_frame()
    out = ConditionNormalizer(n_conditions=2).fit_transform(df)
    for _, g in out.groupby("condition"):
        assert np.allclose(g["s_2"].mean(), 0.0, atol=1e-9)
        assert np.allclose(g["s_2"].std(ddof=0), 1.0, atol=1e-9)


def test_constant_sensor_is_dropped():
    norm = ConditionNormalizer(n_conditions=2).fit(_two_condition_frame())
    assert "s_1" not in norm.sensors_
    assert "s_2" in norm.sensors_


def test_transform_uses_frozen_training_statistics():
    """Leakage guard: shifting new data must show up as a shift, not be re-normalised away."""
    norm = ConditionNormalizer(n_conditions=2).fit(_two_condition_frame(seed=0))
    shifted = norm.transform(_two_condition_frame(shift=5.0, seed=1))
    assert shifted["s_2"].mean() > 3.0


def test_without_condition_normalisation_regime_dominates():
    """With one global scaler the regime switch is the biggest signal in the data."""
    df = _two_condition_frame()
    glob = ConditionNormalizer(n_conditions=1).fit_transform(df)
    per = ConditionNormalizer(n_conditions=2).fit_transform(df)
    assert glob["s_2"].abs().mean() > 0.9  # values sit near +-1 regardless of noise
    assert per["s_2"].abs().mean() < glob["s_2"].abs().mean()


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ConditionNormalizer().transform(_two_condition_frame())


def _series() -> pd.DataFrame:
    return pd.DataFrame(
        {"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2], "x": [1.0, 2.0, 3.0, 10.0, 20.0]}
    )


def test_rolling_features_are_causal_and_per_unit():
    out = add_rolling_features(_series(), ["x"], window=2)
    assert out["x_mean2"].tolist() == [1.0, 1.5, 2.5, 10.0, 15.0]  # unit 2 starts fresh
    assert out["x_std2"].iloc[0] == 0.0


def test_rolling_features_ignore_the_future():
    a = add_rolling_features(_series(), ["x"], window=3)
    future_changed = _series()
    future_changed.loc[2, "x"] = 999.0  # change unit 1, cycle 3
    b = add_rolling_features(future_changed, ["x"], window=3)
    assert a.loc[:1, "x_mean3"].equals(b.loc[:1, "x_mean3"])
