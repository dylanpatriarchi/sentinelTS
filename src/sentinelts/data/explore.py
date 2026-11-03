"""Dataset statistics that motivate the evaluation protocol (see docs/evaluation_protocol.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentinelts.data.cmapss import OPERATIONAL_SETTINGS, SENSORS, load_subset
from sentinelts.data.labels import ANOMALOUS, add_anomaly_labels


def describe_subset(
    root: Path | str, subset: str, anomaly_rul: int = 30, healthy_rul: int = 100
) -> dict[str, Any]:
    train = load_subset(root, subset, "train")
    test = load_subset(root, subset, "test")
    life = train.groupby("unit")["cycle"].max()
    test_len = test.groupby("unit")["cycle"].max()
    test_end_rul = test.groupby("unit")["rul"].min()
    std = train[SENSORS].std()
    test_labels = add_anomaly_labels(test, anomaly_rul, healthy_rul)
    train_labels = add_anomaly_labels(train, anomaly_rul, healthy_rul)["label"]

    def mmm(s) -> tuple[int, int, int]:
        return int(s.min()), int(s.median()), int(s.max())

    return {
        "subset": subset,
        "train_engines": int(life.size),
        "train_lifetime_min_median_max": mmm(life),
        "test_engines": int(test_len.size),
        "test_length_min_median_max": mmm(test_len),
        "test_final_rul_min_median_max": mmm(test_end_rul),
        "operating_conditions": len(train[OPERATIONAL_SETTINGS].round(0).drop_duplicates()),
        "near_constant_sensors": std[std < 1e-3].index.tolist(),
        "train_label_counts": {int(k): int(v) for k, v in train_labels.value_counts().items()},
        "test_engines_reaching_anomaly": int(
            test_labels.groupby("unit")["label"].max().eq(ANOMALOUS).sum()
        ),
    }
