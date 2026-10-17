"""Price elasticity of demand: what the observational data can and cannot say.

A regression of log weekly bookings on log ADR with fixed effects gives the elasticity *implied by past
pricing*. In this data it comes out positive: the hotels raise prices when they expect demand, so price and
bookings move together (simultaneity). Adding controls for the demand the hotel could see (week of year, the
other hotel's bookings the same week as a market proxy, lead time) narrows but cannot close that gap. So the
module reports three things honestly: the naive estimate, the controlled estimate, and the planning
assumption the pricing test and the price endpoint actually use, with a sensitivity range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PLANNING_ELASTICITY = -1.0
SENSITIVITY = (-0.5, -1.0, -1.5)
PLANNING_NOTE = (
    "Planning assumption, not an estimate from this data: unit elasticity (revenue flat to small price changes), "
    "bracketed by -0.5 and -1.5 in the sensitivity analysis. The A/B test exists to replace it with a measurement."
)


def _fit(panel: pd.DataFrame, formula: str) -> dict:
    model = smf.ols(formula, data=panel).fit(cov_type="HC3")
    coef = float(model.params["np.log(adr_mean)"])
    ci = model.conf_int().loc["np.log(adr_mean)"]
    return {
        "elasticity": coef,
        "se": float(model.bse["np.log(adr_mean)"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(model.pvalues["np.log(adr_mean)"]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }


def add_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Week of year, the other hotel's total bookings in the same week (market demand proxy), mean lead time."""
    p = panel[(panel["bookings"] > 0) & (panel["adr_mean"] > 0)].copy()
    p["week_of_year"] = p["week"].dt.isocalendar().week.astype(int)
    weekly_hotel = p.groupby(["hotel", "week"])["bookings"].sum().rename("hotel_week_bookings").reset_index()
    market = weekly_hotel.pivot(index="week", columns="hotel", values="hotel_week_bookings")
    other = {}
    for hotel in market.columns:
        others = market.drop(columns=hotel).sum(axis=1)
        other[hotel] = others
    p["other_hotel_bookings"] = [other[h].get(w, np.nan) for h, w in zip(p["hotel"], p["week"], strict=True)]
    p["other_hotel_bookings"] = p["other_hotel_bookings"].fillna(p["other_hotel_bookings"].median()).clip(lower=1)
    if "lead_time_mean" not in p.columns:
        p["lead_time_mean"] = 0.0
    return p


NAIVE = "np.log(bookings) ~ np.log(adr_mean) + C(hotel) + C(market_segment) + C(month) + C(year)"
CONTROLLED = (
    "np.log(bookings) ~ np.log(adr_mean) + C(hotel) + C(market_segment) + C(week_of_year) + C(year) "
    "+ np.log(other_hotel_bookings) + lead_time_mean"
)


def estimate(panel: pd.DataFrame, min_rows: int = 40) -> dict:
    p = add_controls(panel)
    naive = _fit(p, NAIVE)
    controlled = _fit(p, CONTROLLED)
    by_hotel = {}
    for hotel, sub in p.groupby("hotel"):
        if len(sub) >= min_rows:
            by_hotel[hotel] = _fit(sub, CONTROLLED.replace("C(hotel) + ", ""))
    by_segment = {}
    for seg, sub in p.groupby("market_segment"):
        if len(sub) >= min_rows:
            by_segment[seg] = _fit(sub, CONTROLLED.replace("C(market_segment) + ", ""))
    identified = controlled["ci_high"] < 0
    return {
        "naive": naive,
        "controlled": controlled,
        "by_hotel": by_hotel,
        "by_segment": by_segment,
        "identified": bool(identified),
        "planning_elasticity": controlled["elasticity"] if identified else PLANNING_ELASTICITY,
        "planning_source": "controlled estimate" if identified else "assumption",
        "sensitivity": list(SENSITIVITY),
        "method": {
            "naive": "OLS of log weekly bookings on log mean ADR with hotel, segment, month and year fixed effects; HC3 errors",
            "controlled": "adds week-of-year effects, the other hotel's bookings that week (log) as a market-demand proxy, and mean lead time",
            "unit": "hotel x market segment x ISO week with at least one non-cancelled priced booking",
        },
        "diagnosis": (
            "A positive or non-negative coefficient means prices and bookings rise together: the hotels set ADR with the "
            "demand they expect, so the regression recovers the pricing rule, not the demand curve (simultaneity). "
            "Controls for the demand the hotel could observe shrink the bias but cannot remove it without an instrument "
            "or an experiment."
        ),
        "planning_note": PLANNING_NOTE,
    }


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "elasticity.json"
    mp = out_dir / "elasticity.md"
    jp.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def row(label: str, r: dict) -> str:
        return f"| {label} | {r['elasticity']:.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['p_value']:.3f} | {r['n']} | {r['r2']:.2f} |"

    lines = [
        "# Price elasticity: what the booking data says",
        "",
        f"Unit of analysis: {result['method']['unit']}.",
        "",
        "| Specification | Elasticity | 95% CI | p | rows | R² |",
        "|---|---|---|---|---|---|",
        row("Naive (hotel, segment, month, year effects)", result["naive"]),
        row("Controlled (+ week of year, other hotel's demand, lead time)", result["controlled"]),
    ]
    lines += [row(f"controlled, {h}", r) for h, r in result["by_hotel"].items()]
    lines += [row(f"controlled, segment {s}", r) for s, r in result["by_segment"].items()]
    lines += [
        "",
        f"**Diagnosis.** {result['diagnosis']}",
        "",
        f"**What the pricing work uses.** Elasticity {result['planning_elasticity']:.2f} ({result['planning_source']}). {result['planning_note']}",
    ]
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp


def load_elasticity(path: Path, hotel: str | None = None, default: float = PLANNING_ELASTICITY) -> tuple[float, str]:
    """The elasticity the pricing endpoint should use and where it comes from."""
    if not path.exists():
        return default, "default assumption (no elasticity report)"
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("identified")
        and hotel
        and hotel in data.get("by_hotel", {})
        and data["by_hotel"][hotel]["ci_high"] < 0
    ):
        return float(data["by_hotel"][hotel]["elasticity"]), f"controlled estimate for {hotel}"
    return float(data.get("planning_elasticity", default)), str(data.get("planning_source", "assumption"))


def implied_demand_multiplier(price_change: float, elasticity: float) -> float:
    """Demand multiplier for a relative price change under constant elasticity: (1 + dp) ** e."""
    return float(np.power(1.0 + price_change, elasticity))
