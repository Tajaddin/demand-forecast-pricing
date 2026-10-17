"""The retrain job: refit the champion on the newest window, score it on a holdout, check the newest data for
drift against the training window, compare with what is in production, and promote only when both gates
pass. Every run writes reports/retrain-latest.json with the decision and its reasons.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dfp.config import Settings
from dfp.data import HOTELS, hotel_series
from dfp.drift import GATE_COLUMNS, drift_frame, run_drift
from dfp.metrics import wape
from dfp.tracking import fit_demand_model, log_and_register, production_version, set_production


def champion_from_report(reports_dir: Path, fallback: str = "seasonal_naive") -> str:
    p = reports_dir / "backtest.json"
    if not p.exists():
        return fallback
    return str(json.loads(p.read_text(encoding="utf-8")).get("champion", fallback))


def load_residuals(reports_dir: Path) -> pd.DataFrame | None:
    p = reports_dir / "backtest-residuals.csv"
    return pd.read_csv(p) if p.exists() else None


def holdout_wape(
    daily: pd.DataFrame, model_type: str, holdout_days: int, residuals: pd.DataFrame | None
) -> tuple[float, dict[str, float]]:
    """Fits on everything but the last `holdout_days`, forecasts them, and scores WAPE pooled and per hotel."""
    last = daily["arrival_date"].max()
    through = last - pd.Timedelta(days=holdout_days)
    model = fit_demand_model(daily, model_type, through, residuals)
    errors, actuals, per_hotel = [], [], {}
    for hotel in HOTELS:
        y = hotel_series(daily, hotel)
        actual = y[y.index > through].to_numpy(dtype=float)
        yhat = model.forecasters[hotel].predict(len(actual))
        per_hotel[hotel] = wape(actual, yhat)
        errors.append(np.abs(actual - yhat).sum())
        actuals.append(np.abs(actual).sum())
    return float(sum(errors) / sum(actuals) * 100), per_hotel


def retrain(
    daily: pd.DataFrame,
    s: Settings,
    force_promote: bool = False,
    simulate_drift: bool = False,
    model_type: str | None = None,
) -> dict:
    reports = s.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    model_type = model_type or champion_from_report(reports)
    residuals = load_residuals(reports)
    last = daily["arrival_date"].max()
    current_start = last - pd.Timedelta(days=s.holdout_days - 1)
    # Seasonally matched reference: the same calendar window one year earlier, widened by two weeks on each
    # side. Comparing the newest weeks with a whole year would flag ordinary seasonality as drift.
    reference_start = current_start - pd.Timedelta(days=365 + 14)
    reference_end = last - pd.Timedelta(days=365 - 14)

    current = daily[daily["arrival_date"] >= current_start].copy()
    reference = daily[(daily["arrival_date"] >= reference_start) & (daily["arrival_date"] <= reference_end)].copy()
    if simulate_drift:
        # A synthetic shock for demonstrating the gate: demand up 60%, prices up 30% in the newest window.
        current["arrivals"] *= 1.6
        current["room_nights"] *= 1.6
        current["revenue"] *= 1.6 * 1.3
        current["adr_mean"] *= 1.3
    drift = run_drift(drift_frame(reference), drift_frame(current), reports, name="drift-latest")

    pooled, per_hotel = holdout_wape(daily, model_type, s.holdout_days, residuals)
    prod = production_version(s)
    prod_wape = float(prod["tags"]["holdout_wape"]) if prod and "holdout_wape" in prod["tags"] else None

    reasons = []
    # The gate judges the columns the forecaster consumes; price and revenue drift is reported for a human.
    gate_flags = [drift["per_column"][c]["drifted"] for c in GATE_COLUMNS if c in drift["per_column"]]
    gate_share = float(np.mean(gate_flags)) if gate_flags else 0.0
    drift_ok = gate_share < s.drift_share_threshold
    if not drift_ok:
        reasons.append(
            f"drift: {gate_share:.0%} of the model's input columns drifted (threshold {s.drift_share_threshold:.0%})"
        )
    quality_ok = prod_wape is None or pooled <= prod_wape * (1 + s.promotion_tolerance)
    if not quality_ok:
        reasons.append(
            f"holdout WAPE {pooled:.1f}% is worse than production {prod_wape:.1f}% by more than {s.promotion_tolerance:.0%}"
        )
    promote = force_promote or (drift_ok and quality_ok)
    if force_promote and reasons:
        reasons.append("promoted anyway: --force")

    model = fit_demand_model(daily, model_type, None, residuals)
    tags = {
        "holdout_wape": f"{pooled:.4f}",
        "drifted_share": f"{drift['drifted_share']:.4f}",
        "gate_drifted_share": f"{gate_share:.4f}",
        "promoted": str(promote).lower(),
        "trained_through": model.trained_through,
        "model_type": model_type,
    }
    version = log_and_register(
        model,
        s,
        run_name=f"retrain-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        params={
            "model_type": model_type,
            "holdout_days": s.holdout_days,
            "simulate_drift": simulate_drift,
            "force": force_promote,
        },
        metrics={
            "holdout_wape": pooled,
            "drifted_share": drift["drifted_share"],
            "gate_drifted_share": gate_share,
            **{f"holdout_wape_{h.split()[0].lower()}": v for h, v in per_hotel.items()},
        },
        tags=tags,
    )
    if promote:
        set_production(s, version)
    outcome = {
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "model_type": model_type,
        "version": version,
        "promoted": promote,
        "reasons": reasons or ["drift below threshold and holdout WAPE within tolerance"],
        "holdout_days": s.holdout_days,
        "holdout_wape": pooled,
        "holdout_wape_by_hotel": per_hotel,
        "production_wape_before": prod_wape,
        "production_version_before": prod["version"] if prod else None,
        "drift": {
            "gate_columns": GATE_COLUMNS,
            "gate_drifted_share": gate_share,
            **{
                k: drift[k]
                for k in ("drifted_share", "drifted_columns", "per_column", "reference_rows", "current_rows")
            },
        },
        "simulate_drift": simulate_drift,
        "trained_through": model.trained_through,
    }
    name = "retrain-blocked-example.json" if simulate_drift else "retrain-latest.json"
    (reports / name).write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    return outcome
