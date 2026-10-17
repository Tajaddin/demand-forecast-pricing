"""Loads the hotel bookings file and builds the series the rest of the project works on.

Daily, per hotel: arrivals (bookings not cancelled), total bookings, room-nights, revenue, mean ADR and
cancellations, on a complete calendar. Weekly, per hotel and market segment: bookings and mean ADR, for the
elasticity model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
HOTELS = ("City Hotel", "Resort Hotel")
# Segments with enough volume and real prices; Complementary (free nights), Aviation and Undefined are dropped.
ELASTICITY_SEGMENTS = ("Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups")
ADR_MIN, ADR_MAX = 1.0, 1000.0  # one booking has ADR 5400 and a few are negative; both are data errors


def load_bookings(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = pd.DataFrame(
        {
            "hotel": raw["hotel"].astype("category"),
            "is_canceled": raw["is_canceled"].astype(int),
            "lead_time": raw["lead_time"].astype(int),
            "arrival_date": pd.to_datetime(
                {
                    "year": raw["arrival_date_year"],
                    "month": raw["arrival_date_month"].map(MONTHS),
                    "day": raw["arrival_date_day_of_month"],
                }
            ),
            "nights": (raw["stays_in_weekend_nights"] + raw["stays_in_week_nights"]).astype(int),
            "guests": (raw["adults"] + raw["children"].fillna(0) + raw["babies"]).astype(int),
            "market_segment": raw["market_segment"].astype("category"),
            "adr": raw["adr"].astype(float),
        }
    )
    return df


def daily_series(bookings: pd.DataFrame) -> pd.DataFrame:
    """One row per hotel per calendar day between the first and last arrival date in the data."""
    b = bookings.copy()
    b["kept"] = 1 - b["is_canceled"]
    valid_price = b["adr"].between(ADR_MIN, ADR_MAX) & (b["nights"] > 0)
    b["revenue"] = np.where((b["kept"] == 1) & valid_price, b["adr"] * b["nights"], 0.0)
    b["room_nights"] = np.where(b["kept"] == 1, b["nights"], 0)
    b["priced"] = np.where((b["kept"] == 1) & valid_price, b["adr"], np.nan)

    grouped = (
        b.groupby(["hotel", "arrival_date"], observed=True)
        .agg(
            arrivals=("kept", "sum"),
            bookings=("kept", "size"),
            cancellations=("is_canceled", "sum"),
            room_nights=("room_nights", "sum"),
            revenue=("revenue", "sum"),
            adr_mean=("priced", "mean"),
            lead_time_mean=("lead_time", "mean"),
        )
        .reset_index()
    )
    start, end = b["arrival_date"].min(), b["arrival_date"].max()
    calendar = pd.MultiIndex.from_product(
        [list(HOTELS), pd.date_range(start, end, freq="D")], names=["hotel", "arrival_date"]
    )
    out = grouped.set_index(["hotel", "arrival_date"]).reindex(calendar).reset_index()
    for col in ("arrivals", "bookings", "cancellations", "room_nights", "revenue"):
        out[col] = out[col].fillna(0).astype(float)
    out["adr_mean"] = out.groupby("hotel", observed=True)["adr_mean"].transform(lambda s: s.ffill().bfill())
    out["lead_time_mean"] = out["lead_time_mean"].fillna(out["lead_time_mean"].mean())
    out["hotel"] = out["hotel"].astype(str)
    return out.sort_values(["hotel", "arrival_date"]).reset_index(drop=True)


def weekly_segment_panel(bookings: pd.DataFrame) -> pd.DataFrame:
    """Hotel x market segment x ISO week: non-cancelled bookings and their mean ADR (for the elasticity model)."""
    b = bookings[
        (bookings["is_canceled"] == 0) & bookings["adr"].between(ADR_MIN, ADR_MAX) & (bookings["nights"] > 0)
    ].copy()
    b = b[b["market_segment"].isin(ELASTICITY_SEGMENTS)]
    b["week"] = b["arrival_date"].dt.to_period("W-SUN").dt.start_time
    panel = (
        b.groupby(["hotel", "market_segment", "week"], observed=True)
        .agg(
            bookings=("adr", "size"),
            adr_mean=("adr", "mean"),
            room_nights=("nights", "sum"),
            lead_time_mean=("lead_time", "mean"),
        )
        .reset_index()
    )
    panel["hotel"] = panel["hotel"].astype(str)
    panel["market_segment"] = panel["market_segment"].astype(str)
    panel["month"] = panel["week"].dt.month
    panel["year"] = panel["week"].dt.year
    return panel.sort_values(["hotel", "market_segment", "week"]).reset_index(drop=True)


@dataclass
class QualityReport:
    rows: int
    hotels: list[str]
    first_day: str
    last_day: str
    days_per_hotel: dict[str, int]
    missing_days: int
    negative_values: int
    adr_out_of_range_bookings: int
    cancelled_share: float
    passed: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def quality_report(bookings: pd.DataFrame, daily: pd.DataFrame) -> QualityReport:
    counts = {str(k): int(v) for k, v in daily.groupby("hotel")["arrival_date"].count().items()}
    expected = (daily["arrival_date"].max() - daily["arrival_date"].min()).days + 1
    missing = int(sum(expected - c for c in counts.values()))
    negative = int((daily[["arrivals", "room_nights", "revenue"]] < 0).to_numpy().sum())
    # zero ADR is a legitimate complementary rate; negative or absurd values are data errors
    out_of_range = int(((bookings["adr"] < 0) | (bookings["adr"] > ADR_MAX)).sum())
    report = QualityReport(
        rows=len(bookings),
        hotels=sorted(counts),
        first_day=str(daily["arrival_date"].min().date()),
        last_day=str(daily["arrival_date"].max().date()),
        days_per_hotel={k: int(v) for k, v in counts.items()},
        missing_days=missing,
        negative_values=negative,
        adr_out_of_range_bookings=out_of_range,
        cancelled_share=float(bookings["is_canceled"].mean()),
        passed=missing == 0 and negative == 0 and set(counts) == set(HOTELS) and out_of_range < 0.01 * len(bookings),
    )
    return report


def hotel_series(daily: pd.DataFrame, hotel: str, column: str = "arrivals") -> pd.Series:
    s = daily.loc[daily["hotel"] == hotel].set_index("arrival_date")[column].astype(float)
    s = s.asfreq("D")
    return s
