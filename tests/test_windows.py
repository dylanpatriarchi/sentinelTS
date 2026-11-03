from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelts.features import make_windows


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "a": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
            "b": [-1.0, -2.0, -3.0, -4.0, -10.0, -20.0],
        }
    )


def test_one_window_per_row_with_expected_shape():
    X = make_windows(_frame(), ["a", "b"], window=3)
    assert X.shape == (6, 3, 2)


def test_window_ends_at_current_cycle_and_is_left_padded():
    X = make_windows(_frame(), ["a", "b"], window=3)
    assert X[3, :, 0].tolist() == [2.0, 3.0, 4.0]  # unit 1, cycle 4
    assert X[0, :, 0].tolist() == [1.0, 1.0, 1.0]  # padded with first observation


def test_windows_never_mix_engines():
    X = make_windows(_frame(), ["a"], window=3)
    assert X[4, :, 0].tolist() == [10.0, 10.0, 10.0]  # unit 2 starts fresh
    assert X[5, :, 0].tolist() == [10.0, 10.0, 20.0]


def test_window_of_one_equals_rows():
    df = _frame()
    X = make_windows(df, ["a", "b"], window=1)
    assert np.array_equal(X[:, 0, :], df[["a", "b"]].to_numpy(dtype=np.float32))


def test_windows_ignore_the_future():
    df = _frame()
    before = make_windows(df, ["a"], window=2)
    df.loc[3, "a"] = 999.0  # change unit 1, last cycle
    after = make_windows(df, ["a"], window=2)
    assert np.array_equal(before[:3], after[:3])


def test_unsorted_input_rejected():
    with pytest.raises(ValueError, match="sorted"):
        make_windows(_frame().iloc[::-1], ["a"], window=2)
