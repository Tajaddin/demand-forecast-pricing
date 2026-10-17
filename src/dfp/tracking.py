"""MLflow: experiment logging, the packaged demand model, the registry and its production alias."""

from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

from dfp.config import Settings
from dfp.data import HOTELS, hotel_series
from dfp.models import Forecaster, make_model


def setup_mlflow(s: Settings) -> None:
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment)


@dataclass
class DemandModel:
    """Per-hotel forecasters of one type plus empirical residual quantiles per horizon step for intervals."""

    model_type: str
    trained_through: str
    forecasters: dict[str, Forecaster]
    last_dates: dict[str, str]
    residual_quantiles: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # hotel -> step -> [q10, q90]
    metrics: dict[str, float] = field(default_factory=dict)

    def forecast(self, hotel: str, horizon: int) -> pd.DataFrame:
        if hotel not in self.forecasters:
            raise KeyError(hotel)
        yhat = self.forecasters[hotel].predict(horizon)
        start = pd.Timestamp(self.last_dates[hotel]) + pd.Timedelta(days=1)
        dates = pd.date_range(start, periods=horizon, freq="D")
        q = self.residual_quantiles.get(hotel, {})
        lo = np.array([q.get(str(min(k, 28)), [0.0, 0.0])[0] for k in range(1, horizon + 1)])
        hi = np.array([q.get(str(min(k, 28)), [0.0, 0.0])[1] for k in range(1, horizon + 1)])
        return pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "yhat": np.round(yhat, 2),
                "lo": np.round(np.clip(yhat + lo, 0, None), 2),
                "hi": np.round(yhat + hi, 2),
            }
        )


class DemandPyfunc(PythonModel):
    """MLflow wrapper: input DataFrame with columns hotel and horizon, output one row per forecast day."""

    def load_context(self, context) -> None:
        with open(context.artifacts["model"], "rb") as fh:
            self.model: DemandModel = pickle.load(fh)  # noqa: S301 (our own artifact)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        frames = []
        for _, row in model_input.iterrows():
            f = self.model.forecast(str(row["hotel"]), int(row["horizon"]))
            f.insert(0, "hotel", str(row["hotel"]))
            frames.append(f)
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["hotel", "date", "yhat", "lo", "hi"])
        )


def residual_quantiles_from_backtest(residuals: pd.DataFrame, model_type: str) -> dict[str, dict[str, list[float]]]:
    """10th and 90th percentile of (actual - forecast) per hotel and step, from the backtest folds."""
    out: dict[str, dict[str, list[float]]] = {}
    sub = residuals[residuals["model"] == model_type]
    for (hotel, step), g in sub.groupby(["hotel", "step"]):
        out.setdefault(str(hotel), {})[str(int(str(step)))] = [
            float(np.percentile(g["error"], 10)),
            float(np.percentile(g["error"], 90)),
        ]
    return out


def fit_demand_model(
    daily: pd.DataFrame, model_type: str, through: pd.Timestamp | None, residuals: pd.DataFrame | None
) -> DemandModel:
    forecasters: dict[str, Forecaster] = {}
    last_dates: dict[str, str] = {}
    for hotel in HOTELS:
        y = hotel_series(daily, hotel)
        if through is not None:
            y = y[y.index <= through]
        forecasters[hotel] = make_model(model_type).fit(y)
        last_dates[hotel] = str(y.index[-1].date())
    return DemandModel(
        model_type=model_type,
        trained_through=max(last_dates.values()),
        forecasters=forecasters,
        last_dates=last_dates,
        residual_quantiles=residual_quantiles_from_backtest(residuals, model_type) if residuals is not None else {},
    )


def log_and_register(
    model: DemandModel,
    s: Settings,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
) -> str:
    """Logs the model to a run and registers a new version; returns the version number."""
    setup_mlflow(s)
    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        pkl = Path(tmp) / "demand_model.pkl"
        pkl.write_bytes(pickle.dumps(model))
        (Path(tmp) / "residual_quantiles.json").write_text(json.dumps(model.residual_quantiles), encoding="utf-8")
        mlflow.log_artifact(str(Path(tmp) / "residual_quantiles.json"))
        example = pd.DataFrame({"hotel": [HOTELS[0]], "horizon": [7]})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DemandPyfunc(),
            artifacts={"model": str(pkl)},
            input_example=example,
            registered_model_name=s.model_name,
            pip_requirements=["pandas", "numpy", "scikit-learn", "statsmodels", "lightgbm", "prophet"],
        )
        version = info.registered_model_version
        client = MlflowClient()
        for k, v in {**tags, "run_id": run.info.run_id}.items():
            client.set_model_version_tag(s.model_name, str(version), k, str(v))
        return str(version)


def set_production(s: Settings, version: str) -> None:
    MlflowClient().set_registered_model_alias(s.model_name, s.model_alias, version)


def production_version(s: Settings) -> dict | None:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(s.model_name, s.model_alias)
    except Exception:
        return None
    return {"version": str(mv.version), "run_id": mv.run_id, "tags": dict(mv.tags), "created": mv.creation_timestamp}


def load_production_model(s: Settings):
    setup_mlflow(s)
    return mlflow.pyfunc.load_model(f"models:/{s.model_name}@{s.model_alias}")
