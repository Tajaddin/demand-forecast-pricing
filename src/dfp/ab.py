"""A simulated A/B pricing test, designed and read out the way a data scientist would for a revenue team.

Design: a switchback test on one hotel. Each day of an 8-week window is randomly assigned to A (current
prices) or B (prices raised by `delta`); the metric is revenue per day. Outcomes are simulated from the
hotel's own recent seasonality, the elasticity estimated from the data, and the empirical day-to-day noise,
so the "truth" is known and the analysis can be checked against it. Everything is seeded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

ALPHA = 0.05
POWER = 0.8


@dataclass
class Baseline:
    hotel: str
    days: int
    weekday_demand: dict[int, float]
    weekday_adr: dict[int, float]
    mean_nights: float
    residual_sd: float
    revenue_per_day_mean: float
    revenue_per_day_sd: float


def baseline_from_history(daily: pd.DataFrame, bookings: pd.DataFrame, hotel: str, lookback_days: int = 56) -> Baseline:
    """Seasonality and noise of the last `lookback_days` days before the test window: the simulator's inputs."""
    h = daily[daily["hotel"] == hotel].sort_values("arrival_date")
    hist = h.iloc[-lookback_days:]
    dow = hist["arrival_date"].dt.dayofweek
    weekday_demand = {int(str(k)): float(v) for k, v in hist.groupby(dow)["arrivals"].mean().items()}
    weekday_adr = {int(str(k)): float(v) for k, v in hist.groupby(dow)["adr_mean"].mean().items()}
    fitted = dow.map(weekday_demand)
    residual_sd = float((hist["arrivals"] - fitted).std(ddof=1))
    kept = bookings[(bookings["hotel"] == hotel) & (bookings["is_canceled"] == 0) & (bookings["nights"] > 0)]
    return Baseline(
        hotel=hotel,
        days=len(hist),
        weekday_demand=weekday_demand,
        weekday_adr=weekday_adr,
        mean_nights=float(kept["nights"].mean()),
        residual_sd=residual_sd,
        revenue_per_day_mean=float(hist["revenue"].mean()),
        revenue_per_day_sd=float(hist["revenue"].std(ddof=1)),
    )


def booking_level_power(
    bookings: pd.DataFrame, hotel: str, days: int, delta: float, elasticity: float, lookback_days: int = 365
) -> dict:
    """The better design: randomise bookings (not days) and measure revenue per booking. With thousands of
    bookings per arm the minimum detectable lift shrinks accordingly."""
    kept = bookings[(bookings["hotel"] == hotel) & (bookings["is_canceled"] == 0) & (bookings["nights"] > 0)].copy()
    kept = kept[kept["arrival_date"] >= kept["arrival_date"].max() - pd.Timedelta(days=lookback_days)]
    rev = (kept["adr"] * kept["nights"]).to_numpy(dtype=float)
    per_day = len(kept) / lookback_days
    n_per_arm = int(per_day * days / 2)
    analysis = TTestIndPower()
    mde_effect = analysis.solve_power(
        nobs1=max(n_per_arm, 2), alpha=ALPHA, power=POWER, ratio=1.0, alternative="two-sided"
    )
    mde_relative = float(mde_effect * rev.std(ddof=1) / rev.mean())
    lift = expected_lift(delta, elasticity)
    effect = abs(lift) * rev.mean() / rev.std(ddof=1)
    power = (
        float(
            analysis.power(effect_size=effect, nobs1=max(n_per_arm, 2), alpha=ALPHA, ratio=1.0, alternative="two-sided")
        )
        if effect > 0
        else ALPHA
    )
    return {
        "bookings_per_arm": n_per_arm,
        "revenue_per_booking_mean": float(rev.mean()),
        "revenue_per_booking_sd": float(rev.std(ddof=1)),
        "mde_relative_at_target_power": mde_relative,
        "expected_revenue_lift": lift,
        "power_at_expected_lift": power,
    }


def expected_lift(delta: float, elasticity: float) -> float:
    """Relative revenue change from a relative price change under constant elasticity: (1+d)^(1+e) - 1."""
    return float((1 + delta) ** (1 + elasticity) - 1)


def power_analysis(baseline: Baseline, days_per_arm: int, delta: float, elasticity: float) -> dict:
    """Analytic two-sample power for the revenue-per-day metric with the historical day-to-day noise."""
    analysis = TTestIndPower()
    sd = baseline.revenue_per_day_sd
    mean = baseline.revenue_per_day_mean
    mde_effect = analysis.solve_power(nobs1=days_per_arm, alpha=ALPHA, power=POWER, ratio=1.0, alternative="two-sided")
    mde_relative = float(mde_effect * sd / mean)
    lift = expected_lift(delta, elasticity)
    effect = lift * mean / sd
    power_at_delta = float(
        analysis.power(effect_size=effect, nobs1=days_per_arm, alpha=ALPHA, ratio=1.0, alternative="two-sided")
    )
    days_needed = (
        analysis.solve_power(effect_size=effect, alpha=ALPHA, power=POWER, ratio=1.0, alternative="two-sided")
        if effect > 0
        else float("inf")
    )
    return {
        "days_per_arm": days_per_arm,
        "alpha": ALPHA,
        "target_power": POWER,
        "revenue_per_day_mean": mean,
        "revenue_per_day_sd": sd,
        "mde_relative_at_target_power": mde_relative,
        "price_change": delta,
        "elasticity": elasticity,
        "expected_revenue_lift": lift,
        "power_at_expected_lift": power_at_delta,
        "days_per_arm_needed_for_expected_lift": float(np.ceil(days_needed)) if np.isfinite(days_needed) else None,
    }


def simulate_test(baseline: Baseline, delta: float, elasticity: float, days: int, seed: int) -> pd.DataFrame:
    """One switchback test: `days` days, each randomly A or B (balanced), outcomes drawn from the baseline."""
    rng = np.random.default_rng(seed)
    arms = np.array(["A"] * (days // 2) + ["B"] * (days - days // 2))
    rng.shuffle(arms)
    start = pd.Timestamp("2017-09-01")  # the first day after the data ends; the calendar only sets weekdays
    rows = []
    for i, arm in enumerate(arms):
        date = start + pd.Timedelta(days=i)
        dow = date.dayofweek
        mu = baseline.weekday_demand[dow]
        adr = baseline.weekday_adr[dow]
        if arm == "B":
            mu = mu * (1 + delta) ** elasticity
            adr = adr * (1 + delta)
        arrivals = max(0.0, np.round(mu + rng.normal(0, baseline.residual_sd)))
        revenue = arrivals * adr * baseline.mean_nights
        rows.append({"date": date, "arm": arm, "arrivals": arrivals, "adr": adr, "revenue": revenue})
    return pd.DataFrame(rows)


def analyze(test: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    a = test.loc[test["arm"] == "A", "revenue"].to_numpy()
    b = test.loc[test["arm"] == "B", "revenue"].to_numpy()
    t = stats.ttest_ind(b, a, equal_var=False)
    diff = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "days_a": len(a),
        "days_b": len(b),
        "revenue_per_day_a": float(a.mean()),
        "revenue_per_day_b": float(b.mean()),
        "difference_per_day": diff,
        "relative_lift": float(diff / a.mean()),
        "ci95_difference": [float(lo), float(hi)],
        "ci95_relative": [float(lo / a.mean()), float(hi / a.mean())],
        "welch_t": float(t.statistic),
        "p_value": float(t.pvalue),
        "arrivals_per_day_a": float(test.loc[test["arm"] == "A", "arrivals"].mean()),
        "arrivals_per_day_b": float(test.loc[test["arm"] == "B", "arrivals"].mean()),
        "significant_at_5pct": bool(t.pvalue < ALPHA),
    }


def monte_carlo_power(baseline: Baseline, delta: float, elasticity: float, days: int, n_sims: int, seed: int) -> dict:
    """Empirical power: the share of simulated tests whose Welch test rejects at 5%, with the true effect known."""
    rejections = 0
    lifts = []
    for i in range(n_sims):
        test = simulate_test(baseline, delta, elasticity, days, seed + i)
        a = test.loc[test["arm"] == "A", "revenue"]
        b = test.loc[test["arm"] == "B", "revenue"]
        p = stats.ttest_ind(b, a, equal_var=False).pvalue
        rejections += int(p < ALPHA)
        lifts.append(b.mean() / a.mean() - 1)
    return {
        "simulations": n_sims,
        "empirical_power": rejections / n_sims,
        "mean_estimated_lift": float(np.mean(lifts)),
        "true_lift": expected_lift(delta, elasticity),
    }


def write_readout(
    baseline: Baseline,
    design: dict,
    design_big: dict,
    result: dict,
    mc: dict,
    mc_big: dict,
    out_dir: Path,
    delta: float,
    delta_big: float,
    sensitivity: list[dict] | None = None,
    booking_level: dict | None = None,
    elasticity_source: str = "assumption",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sensitivity = sensitivity or []
    payload = {
        "baseline": asdict(baseline),
        "design": design,
        "design_larger_change": design_big,
        "result": result,
        "monte_carlo": mc,
        "monte_carlo_larger_change": mc_big,
        "sensitivity": sensitivity,
        "booking_level_design": booking_level,
        "elasticity_source": elasticity_source,
    }
    jp = out_dir / "ab-test.json"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mp = out_dir / "ab-readout.md"
    sig = result["significant_at_5pct"]
    lo, hi = result["ci95_relative"]
    decision = (
        "Keep the higher price: the lift is positive and the interval excludes zero."
        if sig and result["relative_lift"] > 0
        else "Do not change prices on this evidence; the test could not tell the lift from zero."
    )
    lines = [
        f"# A/B pricing test readout: {baseline.hotel}, +{delta:.0%} on treatment days",
        "",
        "**Simulated test.** Outcomes come from the hotel's own recent seasonality, the elasticity estimated from the",
        "data and its day-to-day noise, so the true effect is known; this is a rehearsal of the design and the analysis,",
        "not an experiment that was run with guests.",
        "",
        "## Question",
        "",
        f"Does raising prices by {delta:.0%} increase revenue per day at the {baseline.hotel}?",
        "",
        "## Design",
        "",
        f"- Switchback: each of {design['days_per_arm'] * 2} days is randomly A (current prices) or B (+{delta:.0%}); {design['days_per_arm']} days per arm.",
        f"- Metric: revenue per day. Over the last {baseline.days} days it averaged {design['revenue_per_day_mean']:,.0f} with a day-to-day SD of {design['revenue_per_day_sd']:,.0f}.",
        f"- Elasticity used for planning: {design['elasticity']:.2f} ({elasticity_source}); under it a +{delta:.0%} price moves revenue by {design['expected_revenue_lift']:+.1%}.",
        f"- Power: with {design['days_per_arm']} days per arm the smallest lift detectable at 80% power is {design['mde_relative_at_target_power']:.1%}; the expected lift gives only {design['power_at_expected_lift']:.0%} power"
        + (
            f", and detecting it reliably would need about {design['days_per_arm_needed_for_expected_lift']:.0f} days per arm."
            if design["days_per_arm_needed_for_expected_lift"]
            else "."
        ),
        f"- Monte Carlo check ({mc['simulations']} simulated tests): empirical power {mc['empirical_power']:.0%}, mean estimated lift {mc['mean_estimated_lift']:+.1%} against a true {mc['true_lift']:+.1%}.",
        "",
        "## Result",
        "",
        f"- Revenue per day: A {result['revenue_per_day_a']:,.0f}, B {result['revenue_per_day_b']:,.0f}; difference {result['difference_per_day']:+,.0f} ({result['relative_lift']:+.1%}).",
        f"- 95% bootstrap CI for the relative lift: [{lo:+.1%}, {hi:+.1%}]; Welch t = {result['welch_t']:.2f}, p = {result['p_value']:.3f}.",
        f"- Arrivals per day: A {result['arrivals_per_day_a']:.1f}, B {result['arrivals_per_day_b']:.1f}.",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Sensitivity to the elasticity assumption",
        "",
        f"| Elasticity | Expected lift from +{delta:.0%} | Power with {design['days_per_arm']} days per arm |",
        "|---|---|---|",
        *[
            f"| {s['elasticity']:.2f} | {s['expected_revenue_lift']:+.1%} | {s['power_at_expected_lift']:.0%} |"
            for s in sensitivity
        ],
        "",
        "## A design that can actually answer the question",
        "",
        *(
            [
                f"Randomising bookings instead of days and measuring revenue per booking gives about {booking_level['bookings_per_arm']:,} bookings per arm in the same 8 weeks "
                f"(mean {booking_level['revenue_per_booking_mean']:,.0f}, SD {booking_level['revenue_per_booking_sd']:,.0f}): the minimum detectable lift at 80% power drops to "
                f"**{booking_level['mde_relative_at_target_power']:.1%}**, and the +{delta:.0%} change under the planning elasticity would be detected with {booking_level['power_at_expected_lift']:.0%} power. "
                "It needs price randomisation at quote time (per booking request), which the switchback does not."
            ]
            if booking_level
            else ["(booking-level design not computed)"]
        ),
        "",
        "## What a larger change would look like",
        "",
        f"With +{delta_big:.0%} instead, the expected lift is {design_big['expected_revenue_lift']:+.1%} and the same 8 weeks give {design_big['power_at_expected_lift']:.0%} analytic power ({mc_big['empirical_power']:.0%} empirical over {mc_big['simulations']} simulations).",
        "",
        "## Caveats",
        "",
        "- The elasticity is a planning assumption: the observational estimate from this data is not identified (prices and bookings rise together because the hotels price to expected demand; see reports/elasticity.md).",
        "- Switchback days are not independent of each other: bookings are made ahead of the stay, so a price seen on day t influences demand on later days. A real test should randomise by booking date or by comparable units.",
        "- Revenue per day ignores cancellations after booking and ancillary spend.",
        "- Simulated noise is normal around a weekday profile; real demand has fatter tails (events, holidays).",
        "",
        "## Next steps",
        "",
        "1. Run the real test for the number of days the power analysis asks for, randomising by booking date.",
        "2. Pre-register the metric, the stopping rule and the analysis (this document).",
        "3. Re-estimate elasticity from the experiment and feed it to the price recommendation endpoint.",
    ]
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp
