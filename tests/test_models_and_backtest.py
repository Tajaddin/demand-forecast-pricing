import numpy as np
import pandas as pd
import pytest

from dfp.backtest import fold_origins, run_backtest, summarise, write_reports
from dfp.data import HOTELS, hotel_series
from dfp.models import ETS, LightGBMDirect, Naive, SeasonalNaive, make_future_features, make_supervised
from dfp.tracking import residual_quantiles_from_backtest


def test_baselines_repeat_history():
    idx = pd.date_range("2024-01-01", periods=21, freq="D")
    y = pd.Series(np.arange(21, dtype=float), index=idx)
    assert np.all(Naive().fit(y).predict(5) == 20)
    assert list(SeasonalNaive().fit(y).predict(9)) == [14, 15, 16, 17, 18, 19, 20, 14, 15]


def test_supervised_frame_matches_future_features():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    y = pd.Series(np.sin(np.arange(120) / 3) * 10 + 50, index=idx)
    x, target = make_supervised(y, max_horizon=7)
    assert len(target) == len(x)
    assert {"lag_1", "lag_28", "roll7", "roll28", "same_dow_mean", "k", "dow", "month", "doy_sin", "doy_cos"} <= set(
        x.columns
    )
    fut = make_future_features(y, 7)
    assert list(fut.columns) == list(x.columns)
    assert fut.index[0] == idx[-1] + pd.Timedelta(days=1)
    # lag_1 at prediction time is the last observed value
    assert fut["lag_1"].iloc[0] == y.iloc[-1]


def test_lightgbm_and_ets_forecast_the_right_shape(daily):
    y = hotel_series(daily, "City Hotel")
    for model in (LightGBMDirect(max_horizon=14), ETS()):
        f = model.fit(y).predict(14)
        assert f.shape == (14,)
        assert np.all(f >= 0)
        assert np.isfinite(f).all()


def test_fold_origins_end_at_the_last_observation():
    origins = fold_origins(n=500, n_folds=4, step=10, max_horizon=28)
    assert origins == [442, 452, 462, 472]
    assert origins[-1] + 28 == 500


def test_backtest_scores_every_model_and_picks_a_champion(daily, tmp_path):
    models = [Naive(), SeasonalNaive(), LightGBMDirect(max_horizon=14)]
    out = run_backtest(daily, horizons=(7, 14), n_folds=3, step=7, models=models, log=lambda m: None)
    assert set(out.results["model"]) == {"naive", "seasonal_naive", "lightgbm"}
    assert len(out.results) == 2 * 3 * 3 * 2  # hotels x folds x models x horizons
    assert len(out.residuals) == 2 * 3 * 3 * 14
    summary = summarise(out.results, main_horizon=14)
    assert summary["champion"] in {"naive", "seasonal_naive", "lightgbm"}
    pooled = pd.DataFrame(summary["pooled"])
    sn = pooled[pooled["model"] == "seasonal_naive"]
    assert np.allclose(sn["wape_vs_snaive"].to_numpy(dtype=float), 1.0)
    jp, mp = write_reports(
        out,
        summary,
        tmp_path,
        {
            "n_folds": 3,
            "step_days": 7,
            "horizons": [7, 14],
            "first_day": "x",
            "last_day": "y",
            "first_origin": "a",
            "last_origin": "b",
            "run_utc": "t",
            "models": [],
        },
    )
    assert jp.exists() and mp.exists() and (tmp_path / "backtest-residuals.csv").exists()
    assert "seasonal naive" in mp.read_text(encoding="utf-8")


def test_a_failing_model_does_not_stop_the_backtest(daily, tmp_path):
    class Broken:
        name = "broken"

        def fit(self, y):
            raise RuntimeError("boom")

        def predict(self, horizon):
            return np.zeros(horizon)

    out = run_backtest(daily, horizons=(7,), n_folds=2, step=7, models=[SeasonalNaive(), Broken()], log=lambda m: None)
    broken = out.results[out.results["model"] == "broken"]
    assert broken["wape"].isna().all()
    assert out.results[out.results["model"] == "seasonal_naive"]["wape"].notna().all()

    meta = {"n_folds": 2, "step_days": 7, "horizons": [7], "first_day": "x", "last_day": "y"}
    _, md = write_reports(
        out, summarise(out.results, main_horizon=7), tmp_path, meta | {"first_origin": "a", "last_origin": "b"}
    )
    text = md.read_text(encoding="utf-8")
    assert "nan" not in text and "| broken |" not in text
    assert "broken failed on every hotel-fold (4): no scores" in text


def test_a_crashed_fold_is_excluded_not_scored_as_perfect(daily):
    class FailsFirstFit:
        """Seasonal naive that crashes on its first fit only: on every other fold it is seasonal naive."""

        name = "flaky"

        def __init__(self):
            self.inner, self.fits = SeasonalNaive(), 0

        def fit(self, y):
            self.fits += 1
            if self.fits == 1:
                raise RuntimeError("boom")
            self.inner.fit(y)
            return self

        def predict(self, horizon):
            return self.inner.predict(horizon)

    out = run_backtest(
        daily, horizons=(7, 14), n_folds=2, step=7, models=[SeasonalNaive(), FailsFirstFit()], log=lambda m: None
    )
    res = out.results
    crashed = (res["model"] == "flaky") & (res["hotel"] == HOTELS[0]) & (res["fold"] == 0)
    assert res.loc[crashed, "failed"].all() and res.loc[~crashed, "failed"].eq(False).all()

    summary = summarise(res, main_horizon=14)
    # with the crashed fold counted as zero error it beat seasonal naive and became the champion
    assert summary["champion"] == "seasonal_naive"
    pooled = pd.DataFrame(summary["pooled"]).set_index(["model", "horizon"])
    assert pooled.loc[("flaky", 14), "failed_folds"] == 1 and pooled.loc[("seasonal_naive", 14), "failed_folds"] == 0
    same_folds = res[
        (res["model"] == "seasonal_naive")
        & (res["horizon"] == 14)
        & ~((res["hotel"] == HOTELS[0]) & (res["fold"] == 0))
    ]
    expected = same_folds["abs_error_sum"].sum() / same_folds["actual_sum"].sum() * 100
    assert pooled.loc[("flaky", 14), "wape_pooled"] == pytest.approx(expected)

    q = residual_quantiles_from_backtest(out.residuals, "flaky")
    assert np.isfinite([v for steps in q.values() for pair in steps.values() for v in pair]).all()
