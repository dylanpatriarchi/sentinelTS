"""Exploratory analysis of C-MAPSS that informs the evaluation protocol.

Prints, for each subset:
- number of engines and lifetime distribution (train) / truncation (test)
- sensors that are (near-)constant and therefore carry no information
- number of distinct operating conditions
- how many test engines actually reach the anomalous zone

Run:  uv run python scripts/explore_dataset.py --root data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sentinelts.data import SENSORS, SUBSETS, add_anomaly_labels, load_subset


def describe(root: Path, subset: str, anomaly_rul: int, healthy_rul: int) -> None:
    train = load_subset(root, subset, "train")
    test = load_subset(root, subset, "test")
    life = train.groupby("unit")["cycle"].max()
    test_len = test.groupby("unit")["cycle"].max()
    test_end_rul = test.groupby("unit")["rul"].min()

    std = train[SENSORS].std()
    constant = std[std < 1e-3].index.tolist()
    n_conditions = len(train[["op_1", "op_2", "op_3"]].round(0).drop_duplicates())

    labelled = add_anomaly_labels(test, anomaly_rul, healthy_rul)
    reaches = labelled.groupby("unit")["label"].max().eq(1).sum()
    tr_lab = add_anomaly_labels(train, anomaly_rul, healthy_rul)["label"].value_counts()

    print(f"\n=== {subset} ===")
    print(
        f"train engines: {life.size}  lifetime min/median/max: "
        f"{life.min()}/{int(life.median())}/{life.max()}"
    )
    print(
        f"test engines:  {test_len.size}  observed length min/median/max: "
        f"{test_len.min()}/{int(test_len.median())}/{test_len.max()}"
    )
    print(
        f"test RUL at last cycle min/median/max: "
        f"{test_end_rul.min()}/{int(test_end_rul.median())}/{test_end_rul.max()}"
    )
    print(f"operating conditions (rounded op settings): {n_conditions}")
    print(f"near-constant sensors (std<1e-3): {constant}")
    print(f"train label counts (0=healthy,-1=ambiguous,1=anomalous): {tr_lab.to_dict()}")
    print(f"test engines reaching RUL<={anomaly_rul}: {reaches}/{test_len.size}")
    print(f"engines shorter than 50 cycles (train): {int(np.sum(life < 50))}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--anomaly-rul", type=int, default=30)
    p.add_argument("--healthy-rul", type=int, default=100)
    args = p.parse_args()
    for subset in SUBSETS:
        describe(args.root, subset, args.anomaly_rul, args.healthy_rul)


if __name__ == "__main__":
    main()
