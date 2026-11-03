"""End-to-end experiment following docs/evaluation_protocol.md.

    train file ─► split engines ─► fit units ─► normaliser.fit(first N cycles)
                                │             └► features ─► detector.fit(first N cycles)
                                └► val units  ─► features ─► scores ─► threshold (healthy val)
                                                                     └► validation metrics
    test file  ─► (only if evaluate_test=True) ─► features ─► scores ─► test metrics

Every fitted object (KMeans, scaler stats, sensor list, detector, threshold) is
learned before val/test rows are touched, and reused unchanged afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelts.data import add_anomaly_labels, load_subset, split_units
from sentinelts.data.labels import HEALTHY
from sentinelts.data.splits import healthy_prefix, select_units
from sentinelts.evaluation import event_metrics, point_metrics, quantile_threshold
from sentinelts.features import ConditionNormalizer, add_rolling_features
from sentinelts.models import AnomalyDetector

N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass(frozen=True)
class ExperimentConfig:
    subset: str = "FD001"
    val_fraction: float = 0.2
    seed: int = 42
    healthy_cycles: int = 50
    anomaly_rul: int = 30
    healthy_rul: int = 100
    threshold_q: float = 0.99
    rolling_window: int = 10  # 0 disables rolling features
    # Cycles 1..warmup_cycles are neither trained on nor scored: rolling features
    # are not yet defined on a full window there (e.g. std of 1 value = 0, which
    # caused spurious alarms at cycle 1). Fixed across configs so that every
    # experiment is evaluated on exactly the same rows.
    warmup_cycles: int = 10

    def __post_init__(self) -> None:
        if self.rolling_window > self.warmup_cycles + 1:
            raise ValueError("rolling_window must be <= warmup_cycles + 1")
        if self.healthy_cycles <= self.warmup_cycles:
            raise ValueError("healthy_cycles must exceed warmup_cycles")


@dataclass
class SplitResult:
    scored: pd.DataFrame  # unit, cycle, rul, label, score, alarm
    point: dict[str, float]
    event: dict[str, float]


@dataclass
class ExperimentResult:
    config: dict
    features: list[str]
    threshold: float
    validation: SplitResult
    test: SplitResult | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out = {"config": self.config, "threshold": self.threshold, "n_features": len(self.features)}
        out["validation"] = {"point": self.validation.point, "event": self.validation.event}
        if self.test is not None:
            out["test"] = {"point": self.test.point, "event": self.test.event}
        return out


class FeaturePipeline:
    """Normaliser + optional causal rolling features. Fitted on training rows only."""

    def __init__(self, n_conditions: int, rolling_window: int, seed: int) -> None:
        self.normalizer = ConditionNormalizer(n_conditions=n_conditions, seed=seed)
        self.rolling_window = rolling_window
        self.feature_names: list[str] | None = None

    def fit(self, train_rows: pd.DataFrame) -> FeaturePipeline:
        self.normalizer.fit(train_rows)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.normalizer.transform(df)
        sensors = list(self.normalizer.sensors_ or [])
        if self.rolling_window > 0:
            out = add_rolling_features(out, sensors, self.rolling_window)
            rolled = [f"{s}_mean{self.rolling_window}" for s in sensors] + [
                f"{s}_std{self.rolling_window}" for s in sensors
            ]
            self.feature_names = sensors + rolled
        else:
            self.feature_names = sensors
        return out


def _evaluate(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, cfg: ExperimentConfig
) -> SplitResult:
    scored = add_anomaly_labels(df[["unit", "cycle", "rul"]], cfg.anomaly_rul, cfg.healthy_rul)
    scored = scored.assign(score=scores, alarm=scores > threshold)
    return SplitResult(
        scored=scored,
        point=point_metrics(scored, threshold).as_dict(),
        event=event_metrics(scored, threshold).as_dict(),
    )


def run_experiment(
    root: Path | str,
    detector_factory: Callable[[], AnomalyDetector],
    cfg: ExperimentConfig | None = None,
    evaluate_test: bool = False,
) -> ExperimentResult:
    cfg = cfg or ExperimentConfig()
    train = load_subset(root, cfg.subset, "train")
    split = split_units(train, cfg.val_fraction, cfg.seed)
    fit_df = select_units(train, split.fit_units)
    val_df = select_units(train, split.val_units)

    def armed(d: pd.DataFrame) -> pd.DataFrame:
        """Rows after the warm-up period, i.e. where the monitor is allowed to alarm."""
        return d[d["cycle"] > cfg.warmup_cycles].reset_index(drop=True)

    # 1. Fit preprocessing on the healthy prefix of fit engines only.
    features = FeaturePipeline(N_CONDITIONS[cfg.subset], cfg.rolling_window, cfg.seed)
    features.fit(healthy_prefix(fit_df, cfg.healthy_cycles))

    # 2. Fit detector on the same rows. Rolling features are causal, so features
    #    of prefix rows only depend on prefix cycles.
    fit_feat = armed(features.transform(fit_df))
    cols = features.feature_names or []
    train_rows = fit_feat[fit_feat["cycle"] <= cfg.healthy_cycles]
    detector = detector_factory().fit(train_rows[cols].to_numpy())

    # 3. Threshold from healthy validation cycles, then validation metrics.
    val_feat = armed(features.transform(val_df))
    val_scores = detector.score(val_feat[cols].to_numpy())
    val_labels = add_anomaly_labels(val_feat, cfg.anomaly_rul, cfg.healthy_rul)["label"]
    threshold = quantile_threshold(val_scores[(val_labels == HEALTHY).to_numpy()], cfg.threshold_q)
    validation = _evaluate(val_feat, val_scores, threshold, cfg)

    result = ExperimentResult(asdict(cfg), cols, threshold, validation)

    # 4. Final hold-out: everything above is frozen.
    if evaluate_test:
        test_feat = armed(features.transform(load_subset(root, cfg.subset, "test")))
        test_scores = detector.score(test_feat[cols].to_numpy())
        result.test = _evaluate(test_feat, test_scores, threshold, cfg)
    return result
