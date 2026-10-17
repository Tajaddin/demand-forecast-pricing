from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfp.data import HOTELS


def synthetic_bookings(days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Bookings with seasonality and a known price response (-0.8) to prices drawn independently of demand,
    so the elasticity estimator has a known answer. The real data is not like that; see ADR 0002."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2016-01-04")  # a Monday
    rows = []
    segments = ["Online TA", "Direct", "Groups", "Corporate", "Offline TA/TO"]
    for hotel, level in zip(HOTELS, (60, 35), strict=True):
        for d in range(days):
            date = start + pd.Timedelta(days=d)
            season = 1 + 0.25 * np.sin(2 * np.pi * d / 365.25) + 0.15 * (date.dayofweek in (4, 5))
            price = 100 * rng.uniform(0.75, 1.25)  # exogenous: not a function of the season
            demand = level * season * (price / 100) ** -0.8
            n = int(rng.poisson(demand))
            for _ in range(n):
                cancelled = int(rng.random() < 0.3)
                rows.append(
                    {
                        "hotel": hotel,
                        "is_canceled": cancelled,
                        "lead_time": int(rng.integers(0, 200)),
                        "arrival_date": date,
                        "nights": int(rng.integers(1, 5)),
                        "guests": 2,
                        "market_segment": rng.choice(segments),
                        "adr": float(price * rng.uniform(0.95, 1.05)),
                    }
                )
    df = pd.DataFrame(rows)
    df["hotel"] = df["hotel"].astype("category")
    df["market_segment"] = df["market_segment"].astype("category")
    return df


@pytest.fixture(scope="session")
def bookings() -> pd.DataFrame:
    return synthetic_bookings()


@pytest.fixture(scope="session")
def daily(bookings: pd.DataFrame) -> pd.DataFrame:
    from dfp.data import daily_series

    return daily_series(bookings)
