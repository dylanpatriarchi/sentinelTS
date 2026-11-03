from __future__ import annotations

import io
import zipfile

import pytest

from sentinelts.data.download import EXPECTED_FILES, extract_cmapss


def _nested_zip(files: list[str]) -> bytes:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name in files:
            z.writestr(name, "1 1 0\n")
        z.writestr("../evil.txt", "should not be extracted")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("6. Turbofan/CMAPSSData.zip", inner.getvalue())
    return outer.getvalue()


def test_extracts_only_expected_files(tmp_path):
    written = extract_cmapss(_nested_zip(EXPECTED_FILES), tmp_path / "raw")
    assert sorted(p.name for p in written) == sorted(EXPECTED_FILES)
    assert not (tmp_path / "evil.txt").exists()


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_cmapss(_nested_zip(EXPECTED_FILES[:-1]), tmp_path / "raw")
