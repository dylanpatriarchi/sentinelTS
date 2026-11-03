"""Download and extract the NASA C-MAPSS dataset with integrity check.

Usage:
    python -m sentinelts.data.download --dest data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
# SHA-256 of the outer zip as downloaded on 2026-11-03. If NASA re-publishes the
# archive this will fail loudly instead of silently changing our data.
CMAPSS_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

EXPECTED_FILES = [f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in range(1, 5)]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_cmapss(outer_zip: bytes, dest: Path) -> list[Path]:
    """Extract the C-MAPSS text files from the (nested) NASA archive into ``dest``.

    The NASA archive contains a folder with an inner ``CMAPSSData.zip``. Only the
    expected ``.txt`` files are written, flattened into ``dest`` (no path traversal).
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(outer_zip)) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner_bytes = outer.read(inner_name)

    written = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for member in inner.namelist():
            name = Path(member).name
            if name in EXPECTED_FILES or name == "readme.txt":
                target = dest / name
                with inner.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                written.append(target)

    missing = set(EXPECTED_FILES) - {p.name for p in written}
    if missing:
        raise FileNotFoundError(f"Archive is missing expected files: {sorted(missing)}")
    return written


def download_cmapss(dest: Path, url: str = CMAPSS_URL, verify: bool = True) -> list[Path]:
    if all((dest / f).exists() for f in EXPECTED_FILES):
        return [dest / f for f in EXPECTED_FILES]
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read()
    if verify and sha256_bytes(payload) != CMAPSS_SHA256:
        raise ValueError("Checksum mismatch for C-MAPSS archive; refusing to extract.")
    return extract_cmapss(payload, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    files = download_cmapss(args.dest)
    print(f"C-MAPSS ready in {args.dest} ({len(files)} files)")


if __name__ == "__main__":
    main()
