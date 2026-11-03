from __future__ import annotations

import numpy as np
import pytest

from sentinelts.data.cmapss import COLUMNS, load_subset


def test_train_rul_counts_down_to_zero(tiny_cmapss):
    df = load_subset(tiny_cmapss, "FD001", "train")
    assert list(df.columns) == [*COLUMNS, "rul"]
    unit2 = df[df.unit == 2]
    assert unit2["rul"].tolist() == [7, 6, 5, 4, 3, 2, 1, 0]
    assert (df.groupby("unit")["rul"].min() == 0).all()


def test_test_rul_uses_nasa_end_of_series_values(tiny_cmapss):
    df = load_subset(tiny_cmapss, "FD001", "test")
    # unit 1: 3 observed cycles, RUL at last cycle = 10 -> [12, 11, 10]
    assert df[df.unit == 1]["rul"].tolist() == [12, 11, 10]
    assert df[df.unit == 2]["rul"].tolist() == [1, 0]


def test_unknown_subset_rejected(tiny_cmapss):
    with pytest.raises(ValueError, match="Unknown subset"):
        load_subset(tiny_cmapss, "FD009", "train")


def test_non_contiguous_cycles_rejected(tiny_cmapss):
    path = tiny_cmapss / "train_FD001.txt"
    data = np.loadtxt(path)
    np.savetxt(path, data[data[:, 1] != 3], fmt="%g")  # drop cycle 3 everywhere
    with pytest.raises(ValueError, match="contiguous"):
        load_subset(tiny_cmapss, "FD001", "train")


def test_rul_file_length_mismatch_rejected(tiny_cmapss):
    (tiny_cmapss / "RUL_FD001.txt").write_text("10\n")
    with pytest.raises(ValueError, match="RUL values"):
        load_subset(tiny_cmapss, "FD001", "test")


@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_real_data_invariants(real_data_root, subset):
    train = load_subset(real_data_root, subset, "train")
    test = load_subset(real_data_root, subset, "test")
    assert (train.groupby("unit")["rul"].min() == 0).all()  # run to failure
    assert (test.groupby("unit")["rul"].min() > 0).all()  # truncated before failure
    assert train[COLUMNS].notna().all().all()
