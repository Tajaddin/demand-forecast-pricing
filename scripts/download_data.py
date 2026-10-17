"""Downloads the hotel booking demand dataset (Antonio, de Almeida and Nunes, Data in Brief 2019, CC BY 4.0)
from the TidyTuesday mirror on GitHub and verifies its size and columns. No account needed.

    python scripts/download_data.py            # -> data/raw/hotels.csv
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "hotels.csv"
EXPECTED_COLUMNS = {
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "market_segment",
    "adr",
    "reservation_status",
}


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 10_000_000:
        print(f"already present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, DEST)
        print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    header = DEST.open(encoding="utf-8").readline().strip().split(",")
    missing = EXPECTED_COLUMNS - set(header)
    if missing:
        print(f"unexpected file: missing columns {sorted(missing)}", file=sys.stderr)
        return 1
    rows = sum(1 for _ in DEST.open(encoding="utf-8")) - 1
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    print(f"{rows:,} rows, {len(header)} columns, sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
