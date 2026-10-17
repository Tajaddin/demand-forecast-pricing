"""Rolling-origin backtest: for each fold the models are refit on everything before the origin and asked for
the next max(horizons) days; errors are scored per horizon window. Every model is compared with seasonal
naive, and the champion is the model with the lowest pooled WAPE at the main horizon that beats seasonal
naive at every horizon. If none does, seasonal naive is the champion.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dfp.data import HOTELS, hotel_series
from dfp.metrics import all_metrics
from dfp.models import Forecaster, available_models

MAIN_HORIZON = 14


@dataclass
class FoldResult:
    hotel: str
    model: str
    fold: int
    origin: str
    horizon: int
    mape: float
    wape: float
    smape: float
    mase: float
    abs_error_sum: float
    actual_sum: float
    fit_seconds: float


@dataclass
class BacktestOutput:
    results: pd.DataFrame
    residuals: pd.DataFrame  # hotel, model, fold, step, actual, forecast, error


def fold_origins(n: int, n_folds: int, step: int, max_horizon: int) -> list[int]:
    """Indices where each fold's test window starts; the last fold ends at the last observation."""
    last_origin = n - max_horizon
    return [last_origin - (n_folds - 1 - i) * step for i in range(n_folds)]


def run_backtest(
    daily: pd.DataFrame,
    horizons: tuple[int, ...] = (7, 14, 28),
    n_folds: int = 12,
    step: int = 14,
    models: list[Forecaster] | None = None,
    hotels: tuple[str, ...] = HOTELS,
    season: int = 7,
    log=print,
) -> BacktestOutput:
    max_h = max(horizons)
    models = models if models is not None else available_models(max_horizon=max_h)
    rows: list[FoldResult] = []
    resid: list[dict] = []
    for hotel in hotels:
        y = hotel_series(daily, hotel)
        origins = fold_origins(len(y), n_folds, step, max_h)
        for fold, origin in enumerate(origins):
            train = y.iloc[:origin]
            actual = y.iloc[origin : origin + max_h].to_numpy(dtype=float)
            for model in models:
                started = time.perf_counter()
                try:
                    yhat = model.fit(train).predict(max_h)
                except Exception as exc:
                    yhat = np.full(max_h, np.nan)
                    log(f"{model.name} failed on {hotel} fold {fold}: {exc}")
                seconds = time.perf_counter() - started
                for step_i, (a, f) in enumerate(zip(actual, yhat, strict=True), start=1):
                    resid.append(
                        {
                            "hotel": hotel,
                            "model": model.name,
                            "fold": fold,
                            "step": step_i,
                            "actual": a,
                            "forecast": f,
                            "error": a - f,
                        }
                    )
                for h in horizons:
                    y_h, yhat_h = actual[:h], yhat[:h]
                    if np.isnan(yhat_h).any():
                        m = {"mape": np.nan, "wape": np.nan, "smape": np.nan, "mase": np.nan}
                    else:
                        m = all_metrics(y_h, yhat_h, train.to_numpy(dtype=float), season)
                    rows.append(
                        FoldResult(
                            hotel=hotel,
                            model=model.name,
                            fold=fold,
                            origin=str(y.index[origin].date()),
                            horizon=h,
                            abs_error_sum=float(np.nansum(np.abs(y_h - yhat_h))),
                            actual_sum=float(np.abs(y_h).sum()),
                            fit_seconds=seconds,
                            **m,
                        )
                    )
            log(f"{hotel} fold {fold + 1}/{n_folds} done")
    return BacktestOutput(results=pd.DataFrame([r.__dict__ for r in rows]), residuals=pd.DataFrame(resid))


def summarise(results: pd.DataFrame, main_horizon: int = MAIN_HORIZON) -> dict:
    """Per hotel x model x horizon: mean fold metrics, pooled WAPE, relative WAPE to seasonal naive."""
    agg = (
        results.groupby(["hotel", "model", "horizon"])
        .agg(
            mape=("mape", "mean"),
            wape_mean=("wape", "mean"),
            smape=("smape", "mean"),
            mase=("mase", "mean"),
            abs_error_sum=("abs_error_sum", "sum"),
            actual_sum=("actual_sum", "sum"),
            folds=("fold", "nunique"),
            fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    agg["wape_pooled"] = agg["abs_error_sum"] / agg["actual_sum"] * 100
    base = agg[agg["model"] == "seasonal_naive"][["hotel", "horizon", "wape_pooled"]].rename(
        columns={"wape_pooled": "snaive_wape"}
    )
    agg = agg.merge(base, on=["hotel", "horizon"])
    agg["wape_vs_snaive"] = agg["wape_pooled"] / agg["snaive_wape"]
    agg["beats_snaive"] = agg["wape_pooled"] < agg["snaive_wape"]

    pooled = (
        results.groupby(["model", "horizon"])
        .agg(
            abs_error_sum=("abs_error_sum", "sum"),
            actual_sum=("actual_sum", "sum"),
            mape=("mape", "mean"),
            mase=("mase", "mean"),
        )
        .reset_index()
    )
    pooled["wape_pooled"] = pooled["abs_error_sum"] / pooled["actual_sum"] * 100
    sn = pooled[pooled["model"] == "seasonal_naive"][["horizon", "wape_pooled"]].rename(
        columns={"wape_pooled": "snaive_wape"}
    )
    pooled = pooled.merge(sn, on="horizon")
    pooled["wape_vs_snaive"] = pooled["wape_pooled"] / pooled["snaive_wape"]
    pooled["beats_snaive"] = pooled["wape_pooled"] < pooled["snaive_wape"]

    beats_all = pooled.groupby("model")["beats_snaive"].all()
    candidates = pooled[(pooled["horizon"] == main_horizon) & pooled["model"].map(beats_all)]
    champion = str(candidates.sort_values("wape_pooled").iloc[0]["model"]) if len(candidates) else "seasonal_naive"
    return {
        "main_horizon": main_horizon,
        "champion": champion,
        "by_hotel": agg.replace({np.nan: None}).to_dict(orient="records"),
        "pooled": pooled.replace({np.nan: None}).to_dict(orient="records"),
    }


def write_reports(output: BacktestOutput, summary: dict, out_dir: Path, meta: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "backtest.json"
    md_path = out_dir / "backtest.md"
    output.results.to_csv(out_dir / "backtest-folds.csv", index=False)
    output.residuals.to_csv(out_dir / "backtest-residuals.csv", index=False)
    json_path.write_text(json.dumps({**meta, **summary}, indent=2, default=str), encoding="utf-8")

    pooled = pd.DataFrame(summary["pooled"])
    lines = [
        f"# Backtest: {meta['n_folds']} rolling-origin folds, step {meta['step_days']} days, horizons {meta['horizons']}",
        "",
        f"Data: daily arrivals per hotel, {meta['first_day']} to {meta['last_day']}; the fold origins run from {meta['first_origin']} to {meta['last_origin']}.",
        f"Champion by pooled WAPE at h={summary['main_horizon']} among models beating seasonal naive at every horizon: **{summary['champion']}**.",
        "",
        "## Pooled over both hotels (WAPE = total absolute error / total arrivals; lower is better)",
        "",
        "| Model | h | WAPE | vs seasonal naive | MAPE | MASE | beats seasonal naive |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in pooled.sort_values(["horizon", "wape_pooled"]).iterrows():
        lines.append(
            f"| {r['model']} | {int(r['horizon'])} | {r['wape_pooled']:.1f}% | {r['wape_vs_snaive']:.2f}x | {r['mape']:.1f}% | {r['mase']:.2f} | {'yes' if r['beats_snaive'] else 'no'} |"
        )
    by_hotel = pd.DataFrame(summary["by_hotel"])
    for hotel in sorted(by_hotel["hotel"].unique()):
        lines += [
            "",
            f"## {hotel}",
            "",
            "| Model | h | WAPE (pooled) | WAPE (mean of folds) | vs seasonal naive | MAPE | MASE | fit s |",
            "|---|---|---|---|---|---|---|---|",
        ]
        sub = by_hotel[by_hotel["hotel"] == hotel].sort_values(["horizon", "wape_pooled"])
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {int(r['horizon'])} | {r['wape_pooled']:.1f}% | {r['wape_mean']:.1f}% | {r['wape_vs_snaive']:.2f}x | {r['mape']:.1f}% | {r['mase']:.2f} | {r['fit_seconds']:.2f} |"
            )
    lines += [
        "",
        "Method: every model is refit on all data before each fold's origin and forecasts the next 28 days; the first 7, 14 and 28",
        "days are scored. MAPE skips days with zero arrivals; WAPE pools absolute errors over the window. MASE scales by the",
        "in-sample seasonal-naive (weekly) error. Nothing is tuned on the test windows.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
