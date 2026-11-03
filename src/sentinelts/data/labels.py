"""Turn a run-to-failure (RUL) dataset into an anomaly-detection benchmark.

C-MAPSS has no "anomaly" labels: it has RUL. We derive three regimes per cycle:

    label =  0  -> healthy     (rul >  healthy_rul)
    label = -1  -> ambiguous   (anomaly_rul < rul <= healthy_rul)  -- excluded from metrics
    label =  1  -> anomalous   (rul <= anomaly_rul)                -- degraded, close to failure

Why a gap? Degradation onset is gradual and differs per engine. Forcing a hard
boundary would penalise a detector for (correctly) noticing early degradation
just because the cycle sits on the "healthy" side of an arbitrary threshold.
Cycles in the gap are ignored when computing point-wise metrics.
"""

from __future__ import annotations

import pandas as pd

HEALTHY, AMBIGUOUS, ANOMALOUS = 0, -1, 1


def add_anomaly_labels(
    df: pd.DataFrame, anomaly_rul: int = 30, healthy_rul: int = 100
) -> pd.DataFrame:
    if "rul" not in df.columns:
        raise KeyError("DataFrame needs a 'rul' column (see sentinelts.data.cmapss)")
    if not 0 <= anomaly_rul <= healthy_rul:
        raise ValueError("Require 0 <= anomaly_rul <= healthy_rul")

    label = pd.Series(AMBIGUOUS, index=df.index, dtype="int8")
    label[df["rul"] > healthy_rul] = HEALTHY
    label[df["rul"] <= anomaly_rul] = ANOMALOUS
    return df.assign(label=label)
