from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sentinelts.data.cmapss import COLUMNS

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def _write_split(path: Path, lengths: dict[int, int], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in lengths.items():
        for cycle in range(1, n + 1):
            rows.append([unit, cycle, *rng.normal(size=len(COLUMNS) - 2).round(4)])
    np.savetxt(path, np.array(rows), fmt="%g")


@pytest.fixture
def tiny_cmapss(tmp_path: Path) -> Path:
    """Fake FD001: 3 train engines (lengths 5,8,4), 2 test engines truncated (3,2)."""
    _write_split(tmp_path / "train_FD001.txt", {1: 5, 2: 8, 3: 4})
    _write_split(tmp_path / "test_FD001.txt", {1: 3, 2: 2}, seed=1)
    (tmp_path / "RUL_FD001.txt").write_text("10\n0\n")
    return tmp_path


@pytest.fixture
def real_data_root() -> Path:
    if not (REAL_DATA / "train_FD001.txt").exists():
        pytest.skip("Real C-MAPSS not downloaded (sentinelts download)")
    return REAL_DATA


def _fleet_rows(lengths: list[int], truncate_rul: list[int] | None, seed: int) -> np.ndarray:
    """Engines whose sensors drift up exponentially in the last ~40 cycles of life."""
    rng = np.random.default_rng(seed)
    rows = []
    for u, life in enumerate(lengths, start=1):
        observed = life - (truncate_rul[u - 1] if truncate_rul else 0)
        for c in range(1, observed + 1):
            rul = life - c
            drift = 3.0 * np.exp(-rul / 15.0)
            ops = [0.0, 0.0, 100.0]
            sensors = rng.normal(size=21) + drift
            rows.append([u, c, *ops, *sensors])
    return np.array(rows)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    train_life = rng.integers(130, 200, size=20).tolist()
    test_life = rng.integers(130, 200, size=6).tolist()
    test_rul = [5, 10, 20, 50, 80, 120]
    np.savetxt(tmp_path / "train_FD001.txt", _fleet_rows(train_life, None, 1), fmt="%.5f")
    np.savetxt(tmp_path / "test_FD001.txt", _fleet_rows(test_life, test_rul, 2), fmt="%.5f")
    (tmp_path / "RUL_FD001.txt").write_text("\n".join(map(str, test_rul)))
    return tmp_path
