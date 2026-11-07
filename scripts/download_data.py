"""Downloads the hotel booking demand dataset (Antonio, de Almeida and Nunes, Data in Brief 2019, CC BY 4.0)
from the TidyTuesday mirror on GitHub, pinned to the commit that added it, and verifies its sha256. No account
needed.

    python scripts/download_data.py            # -> data/raw/hotels.csv
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "d75aaa0d31596ad6487ae0db20138067d301e031"  # "add week 7 - hotels dataset", the only commit to the file
URL = f"https://raw.githubusercontent.com/rfordatascience/tidytuesday/{COMMIT}/data/2020/2020-02-11/hotels.csv"
SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "hotels.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"already present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"downloading {URL}")
        part = DEST.with_name(DEST.name + ".part")
        urllib.request.urlretrieve(URL, part)
        digest = sha256(part)
        if digest != SHA256:
            part.unlink()
            print(f"unexpected file: sha256 {digest}, expected {SHA256}", file=sys.stderr)
            return 1
        part.replace(DEST)
        print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    header = DEST.open(encoding="utf-8").readline().strip().split(",")
    rows = sum(1 for _ in DEST.open(encoding="utf-8")) - 1
    print(f"{rows:,} rows, {len(header)} columns, sha256 {SHA256[:16]} verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
