"""Preprocessing that is fitted on training rows only and then frozen.

Why normalise *per operating condition*?
    In FD002/FD004 the engine flies in 6 regimes (altitude, Mach, throttle).
    Switching regime moves every sensor far more than degradation does, so a
    detector on raw values would mostly detect "the regime changed". We first
    identify the regime from the operational settings (KMeans), then z-score
    each sensor with the mean/std *of that regime in healthy training data*.
    After this, a value of +3 means "3 healthy standard deviations above what
    this engine type normally shows in this regime" -- a deviation from health.

Why fit only on the healthy training prefix?
    Statistics computed on validation/test rows (or on degraded rows) would leak
    information about the data we evaluate on, and would partly "normalise away"
    the degradation we want to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sentinelts.data.cmapss import OPERATIONAL_SETTINGS, SENSORS


class ConditionNormalizer:
    """Cluster operating conditions, drop uninformative sensors, z-score per condition.

    Parameters
    ----------
    n_conditions:
        Number of operating regimes (1 for FD001/FD003, 6 for FD002/FD004).
    min_std:
        A sensor is dropped if its within-condition std on training data is below
        this value in *any* condition (it is constant or quantised there, and
        z-scoring it would divide by ~0). Absolute value: chosen by inspecting
        C-MAPSS (s_6 has std ~0.002, the smallest informative sensor s_15 ~0.027).
    """

    def __init__(self, n_conditions: int = 1, min_std: float = 1e-2, seed: int = 0) -> None:
        if n_conditions < 1:
            raise ValueError("n_conditions must be >= 1")
        self.n_conditions = n_conditions
        self.min_std = min_std
        self.seed = seed
        self._kmeans: KMeans | None = None
        self.sensors_: list[str] | None = None
        self.mean_: pd.DataFrame | None = None
        self.std_: pd.DataFrame | None = None

    # -- condition assignment ------------------------------------------------
    def _conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions == 1:
            return np.zeros(len(df), dtype=int)
        assert self._kmeans is not None
        return self._kmeans.predict(df[OPERATIONAL_SETTINGS].to_numpy())

    # -- sklearn-like API ----------------------------------------------------
    def fit(self, df: pd.DataFrame) -> ConditionNormalizer:
        if self.n_conditions > 1:
            self._kmeans = KMeans(self.n_conditions, n_init=10, random_state=self.seed)
            self._kmeans.fit(df[OPERATIONAL_SETTINGS].to_numpy())
        cond = self._conditions(df)
        grouped = df[SENSORS].groupby(cond)
        if grouped.ngroups != self.n_conditions:
            raise ValueError("Some operating condition has no training rows")
        std = grouped.std(ddof=0)
        self.sensors_ = [s for s in SENSORS if (std[s] >= self.min_std).all()]
        self.mean_ = grouped.mean()[self.sensors_]
        self.std_ = std[self.sensors_]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with ``condition`` and z-scored sensors; dropped sensors removed."""
        if self.sensors_ is None:
            raise RuntimeError("ConditionNormalizer must be fitted before transform")
        assert self.mean_ is not None and self.std_ is not None
        cond = self._conditions(df)
        mu = self.mean_.loc[cond].to_numpy()
        sd = self.std_.loc[cond].to_numpy()
        z = (df[self.sensors_].to_numpy() - mu) / sd
        keep = [c for c in df.columns if c not in SENSORS]
        out = df[keep].copy()
        out["condition"] = cond
        out[self.sensors_] = z
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def add_rolling_features(df: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    """Append causal rolling mean and std of ``columns`` computed *within* each unit.

    Causal: the value at cycle t uses cycles t-window+1 .. t only (never the
    future). Grouping by unit prevents one engine's history bleeding into the
    next engine's first cycles. Early cycles use the available history
    (``min_periods=1``; std of a single value is defined as 0).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if not df["unit"].is_monotonic_increasing:
        raise ValueError("df must be sorted by unit, cycle")
    g = df.groupby("unit", sort=False)[columns]
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(window, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
    mean.columns = [f"{c}_mean{window}" for c in columns]
    std.columns = [f"{c}_std{window}" for c in columns]
    return pd.concat([df, mean.loc[df.index], std.loc[df.index].fillna(0.0)], axis=1)
