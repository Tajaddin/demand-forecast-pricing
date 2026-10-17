"""Forecasters with one interface: fit(y) then predict(horizon). Baselines first; a model that cannot beat
them in the backtest is reported as such rather than hidden.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

SEASON = 7


class Forecaster(Protocol):
    name: str

    def fit(self, y: pd.Series) -> Forecaster: ...

    def predict(self, horizon: int) -> np.ndarray: ...


@dataclass
class Naive:
    name: str = "naive"
    last: float = 0.0

    def fit(self, y: pd.Series) -> Naive:
        self.last = float(y.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last)


@dataclass
class SeasonalNaive:
    name: str = "seasonal_naive"
    m: int = SEASON
    season: np.ndarray = field(default_factory=lambda: np.zeros(SEASON))

    def fit(self, y: pd.Series) -> SeasonalNaive:
        self.season = y.to_numpy(dtype=float)[-self.m :]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / self.m))
        return np.tile(self.season, reps)[:horizon]


@dataclass
class ETS:
    """Additive error, damped additive trend, additive weekly seasonality (statsmodels ETSModel)."""

    name: str = "ets"
    _fit: object = None

    def fit(self, y: pd.Series) -> ETS:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ETSModel(
                y.to_numpy(dtype=float),
                error="add",
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=SEASON,
            )
            self._fit = model.fit(disp=False, maxiter=200)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.clip(np.asarray(self._fit.forecast(horizon), dtype=np.float64), 0, None)  # type: ignore[attr-defined]


@dataclass
class SARIMA:
    """SARIMA(1,0,1)(1,1,1)[7]: weekly seasonal differencing, short memory, no exogenous terms."""

    name: str = "sarima"
    _fit: object = None

    def fit(self, y: pd.Series) -> SARIMA:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y.to_numpy(dtype=float),
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, SEASON),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit = model.fit(disp=False, maxiter=200)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.clip(np.asarray(self._fit.forecast(horizon), dtype=np.float64), 0, None)  # type: ignore[attr-defined]


@dataclass
class ProphetModel:
    name: str = "prophet"
    _model: object = None
    _last: pd.Timestamp | None = None

    def fit(self, y: pd.Series) -> ProphetModel:
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        df = pd.DataFrame({"ds": y.index, "y": y.to_numpy(dtype=float)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, uncertainty_samples=0
            )
            m.fit(df)
        self._model = m
        self._last = pd.Timestamp(y.index[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = pd.DataFrame({"ds": pd.date_range(self._last + pd.Timedelta(days=1), periods=horizon, freq="D")})  # type: ignore[operator]
        out = self._model.predict(future)  # type: ignore[attr-defined]
        return np.clip(out["yhat"].to_numpy(dtype=float), 0, None)


LAGS = tuple(range(1, 29))


def _calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    doy = dates.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "dow": dates.dayofweek.to_numpy(),
            "month": dates.month.to_numpy(),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=dates,
    )


def make_supervised(y: pd.Series, max_horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per (origin t, step k): lags of y up to the origin, calendar features of the target date, and k."""
    vals = y.to_numpy(dtype=float)
    idx = pd.DatetimeIndex(y.index)
    cal = _calendar(idx)
    rows, targets = [], []
    max_lag = max(LAGS)
    for t in range(max_lag, len(vals)):
        lags = vals[
            t - np.array(LAGS)
        ]  # lag_1 = y[t-1] ... (origin t means we forecast y[t + k - 1] from history up to t-1)
        roll7 = vals[t - 7 : t].mean()
        roll28 = vals[t - 28 : t].mean()
        for k in range(1, max_horizon + 1):
            target_pos = t + k - 1
            if target_pos >= len(vals):
                break
            offsets = target_pos - np.arange(1, 5) * 7  # the same weekday, last four occurrences before the origin
            offsets = offsets[offsets < t]
            same_dow = vals[offsets] if len(offsets) else np.array([roll7])
            feat = {f"lag_{lag}": lv for lag, lv in zip(LAGS, lags, strict=True)}
            feat.update(
                {
                    "roll7": roll7,
                    "roll28": roll28,
                    "same_dow_mean": float(same_dow.mean()),
                    "k": k,
                    **{str(k): v for k, v in cal.iloc[target_pos].items()},
                }
            )
            rows.append(feat)
            targets.append(vals[target_pos])
    return pd.DataFrame(rows), np.asarray(targets)


def make_future_features(y: pd.Series, horizon: int) -> pd.DataFrame:
    """Features for the next `horizon` days after the end of y, built the same way as in training."""
    vals = y.to_numpy(dtype=float)
    t = len(vals)
    last = pd.Timestamp(y.index[-1])
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    cal = _calendar(dates)
    lags = vals[t - np.array(LAGS)]
    roll7, roll28 = vals[t - 7 :].mean(), vals[t - 28 :].mean()
    rows = []
    for k in range(1, horizon + 1):
        target_pos = t + k - 1
        offsets = target_pos - np.arange(1, 5) * 7  # the same weekday, last four occurrences in history
        offsets = offsets[offsets < t]
        same = vals[offsets] if len(offsets) else np.array([roll7])
        feat = {f"lag_{lag}": lv for lag, lv in zip(LAGS, lags, strict=True)}
        feat.update(
            {
                "roll7": roll7,
                "roll28": roll28,
                "same_dow_mean": float(same.mean()),
                "k": k,
                **{str(k): v for k, v in cal.iloc[k - 1].items()},
            }
        )
        rows.append(feat)
    return pd.DataFrame(rows, index=dates)


@dataclass
class LightGBMDirect:
    """One gradient-boosting model over (origin, step) rows: lags, rolling means, same-weekday mean, calendar, k."""

    name: str = "lightgbm"
    max_horizon: int = 28
    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 7,
            "verbose": -1,
        }
    )
    _model: object = None
    _y: pd.Series | None = None

    def fit(self, y: pd.Series) -> LightGBMDirect:
        import lightgbm as lgb

        x, target = make_supervised(y, self.max_horizon)
        model = lgb.LGBMRegressor(**self.params)
        model.fit(x, target)
        self._model = model
        self._y = y
        return self

    def predict(self, horizon: int) -> np.ndarray:
        assert self._y is not None
        assert self._model is not None
        x = make_future_features(self._y, horizon)
        return np.clip(np.asarray(self._model.predict(x), dtype=np.float64), 0, None)  # type: ignore[attr-defined]


def available_models(include_prophet: bool = True, max_horizon: int = 28) -> list[Forecaster]:
    models: list[Forecaster] = [Naive(), SeasonalNaive(), ETS(), SARIMA(), LightGBMDirect(max_horizon=max_horizon)]
    if include_prophet:
        try:
            import prophet  # noqa: F401

            models.append(ProphetModel())
        except Exception:
            logging.getLogger(__name__).warning("prophet not importable; skipping it")
    return models


def make_model(name: str, max_horizon: int = 28) -> Forecaster:
    for m in available_models(max_horizon=max_horizon):
        if m.name == name:
            return m
    raise KeyError(name)
