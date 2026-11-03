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
        pytest.skip("Real C-MAPSS not downloaded (python -m sentinelts.data.download)")
    return REAL_DATA
