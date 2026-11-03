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
from sentinelts.features import ConditionNormalizer, add_rolling_features, make_windows
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
    # Causal moving average of the last k anomaly scores of the same engine,
    # applied before thresholding. 1 = raw scores. Model-agnostic, so it can be
    # compared fairly across detectors.
    score_smoothing: int = 1

    def __post_init__(self) -> None:
        if self.score_smoothing < 1:
            raise ValueError("score_smoothing must be >= 1")
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
        out.update(self.extra)
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
        sensors = list(self.normalizer.sensors_ or [])
        w = self.rolling_window
        rolled = [f"{s}_mean{w}" for s in sensors] + [f"{s}_std{w}" for s in sensors]
        self.feature_names = sensors + (rolled if w > 0 else [])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.normalizer.transform(df)
        if self.rolling_window > 0:
            out = add_rolling_features(
                out, list(self.normalizer.sensors_ or []), self.rolling_window
            )
        return out


def smooth_scores(scores: np.ndarray, units: np.ndarray, k: int) -> np.ndarray:
    """Causal per-engine moving average over the last ``k`` scores (fewer at the start)."""
    if k == 1:
        return np.asarray(scores, dtype=float)
    s = pd.Series(np.asarray(scores, dtype=float))
    return (
        s.groupby(units, sort=False)
        .transform(lambda g: g.rolling(k, min_periods=1).mean())
        .to_numpy()
    )


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

    detector = detector_factory()

    def windows_and_rows(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Causal windows for the armed rows (history may reach into the warm-up)."""
        feat = features.transform(df)
        X = make_windows(feat, cols, detector.window)
        armed = (feat["cycle"] > cfg.warmup_cycles).to_numpy()
        return X[armed], feat[armed].reset_index(drop=True)

    # 1. Fit preprocessing on the healthy prefix of fit engines only.
    features = FeaturePipeline(N_CONDITIONS[cfg.subset], cfg.rolling_window, cfg.seed)
    features.fit(healthy_prefix(fit_df, cfg.healthy_cycles))
    cols = features.feature_names or []

    # 2. Fit detector on windows ending in the healthy prefix. Windows and rolling
    #    features are causal, so they only contain prefix cycles.
    X_fit, fit_rows = windows_and_rows(fit_df)
    in_prefix = (fit_rows["cycle"] <= cfg.healthy_cycles).to_numpy()
    detector.fit(X_fit[in_prefix], groups=fit_rows["unit"].to_numpy()[in_prefix])

    # 3. Threshold from healthy validation cycles, then validation metrics.
    X_val, val_rows = windows_and_rows(val_df)
    val_scores = smooth_scores(
        detector.score(X_val), val_rows["unit"].to_numpy(), cfg.score_smoothing
    )
    val_labels = add_anomaly_labels(val_rows, cfg.anomaly_rul, cfg.healthy_rul)["label"]
    threshold = quantile_threshold(val_scores[(val_labels == HEALTHY).to_numpy()], cfg.threshold_q)
    validation = _evaluate(val_rows, val_scores, threshold, cfg)

    result = ExperimentResult(asdict(cfg), cols, threshold, validation)
    if hasattr(detector, "history"):
        result.extra["training_history"] = detector.history

    # 4. Final hold-out: everything above is frozen.
    if evaluate_test:
        X_test, test_rows = windows_and_rows(load_subset(root, cfg.subset, "test"))
        test_scores = smooth_scores(
            detector.score(X_test), test_rows["unit"].to_numpy(), cfg.score_smoothing
        )
        result.test = _evaluate(test_rows, test_scores, threshold, cfg)
    return result
