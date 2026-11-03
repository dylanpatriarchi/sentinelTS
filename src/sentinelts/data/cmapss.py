"""Load NASA C-MAPSS subsets into tidy DataFrames with a ground-truth RUL column.

Each row is one operational cycle of one engine (``unit``). Columns:
``unit, cycle, op_1..op_3, s_1..s_21, rul``.

RUL (Remaining Useful Life) = number of cycles left before failure.
- train: every engine runs to failure, so ``rul = last_cycle - cycle``.
- test: series are truncated before failure; NASA gives the RUL at the last
  observed cycle in ``RUL_FD00x.txt``, so ``rul = rul_at_end + last_cycle - cycle``.

RUL is *ground truth for evaluation only*. It must never be used as a model input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
OPERATIONAL_SETTINGS = [f"op_{i}" for i in range(1, 4)]
SENSORS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OPERATIONAL_SETTINGS, *SENSORS]

Split = Literal["train", "test"]


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    return df.astype({"unit": "int32", "cycle": "int32"})


def _check_monotonic_cycles(df: pd.DataFrame) -> None:
    """Each unit must start at cycle 1 and increase by exactly 1 (no gaps, no shuffling)."""
    diffs = df.groupby("unit")["cycle"].diff().dropna()
    starts = df.groupby("unit")["cycle"].first()
    if not (diffs == 1).all() or not (starts == 1).all():
        raise ValueError("Cycles are not contiguous 1..N within every unit")


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    last = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=(last - df["cycle"]).astype("int32"))


def add_test_rul(df: pd.DataFrame, rul_at_end: np.ndarray) -> pd.DataFrame:
    units = np.sort(df["unit"].unique())
    if len(units) != len(rul_at_end):
        raise ValueError(f"{len(units)} test units but {len(rul_at_end)} RUL values")
    end_rul = pd.Series(rul_at_end, index=units)
    last = df.groupby("unit")["cycle"].transform("max")
    rul = df["unit"].map(end_rul) + last - df["cycle"]
    return df.assign(rul=rul.astype("int32"))


def load_subset(root: Path | str, subset: str, split: Split) -> pd.DataFrame:
    """Load one C-MAPSS subset/split with a ``rul`` column, sorted by (unit, cycle)."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; choose from {SUBSETS}")
    root = Path(root)
    df = _read_table(root / f"{split}_{subset}.txt")
    df = df.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)
    _check_monotonic_cycles(df)

    if split == "train":
        return add_train_rul(df)
    rul_at_end = np.loadtxt(root / f"RUL_{subset}.txt", dtype=int, ndmin=1)
    return add_test_rul(df, rul_at_end)
