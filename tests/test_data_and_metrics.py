import numpy as np
import pytest

from dfp.data import HOTELS, hotel_series, quality_report, weekly_segment_panel
from dfp.metrics import all_metrics, mape, mase, smape, wape


def test_daily_series_has_a_complete_calendar_per_hotel(bookings, daily):
    days = (bookings["arrival_date"].max() - bookings["arrival_date"].min()).days + 1
    assert set(daily["hotel"]) == set(HOTELS)
    assert daily.groupby("hotel")["arrival_date"].count().eq(days).all()
    assert (daily["arrivals"] >= 0).all()
    assert daily["adr_mean"].notna().all()


def test_arrivals_exclude_cancellations_and_revenue_uses_kept_nights(bookings, daily):
    kept = bookings[bookings["is_canceled"] == 0]
    one = kept.iloc[0]
    day = daily[(daily["hotel"] == str(one["hotel"])) & (daily["arrival_date"] == one["arrival_date"])].iloc[0]
    same_day = kept[(kept["hotel"] == one["hotel"]) & (kept["arrival_date"] == one["arrival_date"])]
    assert day["arrivals"] == len(same_day)
    assert day["revenue"] == pytest.approx((same_day["adr"] * same_day["nights"]).sum())


def test_quality_report_passes_on_clean_data(bookings, daily):
    report = quality_report(bookings, daily)
    assert report.passed
    assert report.missing_days == 0
    assert report.days_per_hotel == {h: int(daily[daily["hotel"] == h].shape[0]) for h in HOTELS}


def test_quality_report_flags_a_gap(bookings, daily):
    broken = daily.drop(daily.index[10]).reset_index(drop=True)
    assert quality_report(bookings, broken).missing_days == 1


def test_weekly_panel_keeps_only_priced_kept_bookings(bookings):
    panel = weekly_segment_panel(bookings)
    assert {"hotel", "market_segment", "week", "bookings", "adr_mean", "month", "year"} <= set(panel.columns)
    assert (panel["bookings"] > 0).all()
    assert (panel["adr_mean"] > 0).all()
    assert panel["week"].dt.dayofweek.eq(0).all()


def test_hotel_series_is_daily_float(daily):
    y = hotel_series(daily, "City Hotel")
    assert y.index.freq == "D"
    assert y.dtype == float


def test_metrics_on_known_values():
    y = np.array([10, 20, 30, 40])
    yhat = np.array([12, 18, 33, 36])
    assert wape(y, yhat) == pytest.approx(11 / 100 * 100)
    assert mape(y, yhat) == pytest.approx(np.mean([0.2, 0.1, 0.1, 0.1]) * 100)
    assert smape(y, yhat) > 0
    train = np.array([1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8])
    assert mase(y, yhat, train, m=7) == pytest.approx(np.mean([2, 2, 3, 4]) / 1.0)
    assert set(all_metrics(y, yhat, train)) == {"mape", "wape", "smape", "mase"}


def test_metrics_handle_zero_actuals():
    assert np.isnan(mape(np.zeros(3), np.ones(3)))
    assert np.isnan(wape(np.zeros(3), np.ones(3)))
    assert mape(np.array([0, 10]), np.array([1, 11])) == pytest.approx(10.0)
