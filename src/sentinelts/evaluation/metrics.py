"""Two complementary views of detector quality.

Point-wise metrics (per cycle)
    Treat every labelled cycle as an independent sample; ambiguous cycles
    (label == -1) are dropped. Threshold-free: ROC-AUC and PR-AUC (average
    precision). PR-AUC is the headline number because anomalies are a minority
    class and ROC-AUC looks optimistic under class imbalance.

Event-wise metrics (per engine)
    What an operator actually cares about: did we raise an alarm before the
    engine failed, how early, and how often did we cry wolf on healthy engines?
    We deliberately do NOT use "point-adjust" (marking a whole anomalous segment
    as detected if any point in it is), which is known to inflate F1 so much
    that random scores can look competitive (Kim et al., AAAI 2022).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinelts.data.labels import AMBIGUOUS, ANOMALOUS, HEALTHY


@dataclass(frozen=True)
class PointMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    anomaly_ratio: float  # = PR-AUC of a random detector, the baseline to beat
    n_points: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EventMetrics:
    n_units: int
    n_units_with_anomaly: int
    detection_rate: float  # fraction of degraded engines with >=1 alarm in anomalous zone
    median_lead_time: float  # median RUL at first non-healthy alarm (higher = earlier)
    false_alarm_unit_rate: float  # fraction of engines with >=1 alarm while healthy
    false_alarm_point_rate: float  # alarms / healthy cycles

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate(df: pd.DataFrame, score_col: str) -> None:
    missing = {"unit", "rul", "label", score_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if not np.isfinite(df[score_col].to_numpy(dtype=float)).all():
        raise ValueError("Scores contain NaN or inf")


def point_metrics(df: pd.DataFrame, threshold: float, score_col: str = "score") -> PointMetrics:
    _validate(df, score_col)
    eval_df = df[df["label"] != AMBIGUOUS]
    y = (eval_df["label"] == ANOMALOUS).to_numpy()
    s = eval_df[score_col].to_numpy(dtype=float)
    if y.all() or not y.any():
        raise ValueError("Point metrics need both healthy and anomalous cycles")

    pred = s > threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PointMetrics(
        roc_auc=float(roc_auc_score(y, s)),
        pr_auc=float(average_precision_score(y, s)),
        precision=precision,
        recall=recall,
        f1=f1,
        anomaly_ratio=float(y.mean()),
        n_points=int(y.size),
    )


def event_metrics(df: pd.DataFrame, threshold: float, score_col: str = "score") -> EventMetrics:
    _validate(df, score_col)
    alarm = df[score_col].to_numpy(dtype=float) > threshold
    df = df.assign(_alarm=alarm)

    lead_times, detected, false_alarm_units = [], 0, 0
    n_with_anomaly = 0
    for _, g in df.groupby("unit", sort=True):
        healthy = g["label"] == HEALTHY
        if (g["_alarm"] & healthy).any():
            false_alarm_units += 1
        if not (g["label"] == ANOMALOUS).any():
            continue  # truncated test engine that never reached the anomalous zone
        n_with_anomaly += 1
        if (g["_alarm"] & (g["label"] == ANOMALOUS)).any():
            detected += 1
            first_alarm = g[g["_alarm"] & ~healthy].iloc[0]
            lead_times.append(int(first_alarm["rul"]))

    healthy_mask = df["label"] == HEALTHY
    n_healthy = int(healthy_mask.sum())
    return EventMetrics(
        n_units=int(df["unit"].nunique()),
        n_units_with_anomaly=n_with_anomaly,
        detection_rate=detected / n_with_anomaly if n_with_anomaly else float("nan"),
        median_lead_time=float(np.median(lead_times)) if lead_times else float("nan"),
        false_alarm_unit_rate=false_alarm_units / df["unit"].nunique(),
        false_alarm_point_rate=(
            float((df["_alarm"] & healthy_mask).sum()) / n_healthy if n_healthy else float("nan")
        ),
    )
