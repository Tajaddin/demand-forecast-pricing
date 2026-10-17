"""Renders reports/*.json into the README between markers, so every number in the README traces back to a
command that can be re-run (dfp backtest | elasticity | ab | train | retrain).

    python scripts/render_results.py           # print
    python scripts/render_results.py --write   # update README.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _load(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def backtest_md() -> str:
    b = _load("backtest.json")
    if not b:
        return "(no backtest yet: `make backtest`)\n"
    lines = [
        "| Model | h=7 WAPE | h=14 WAPE | h=28 WAPE | vs seasonal naive (h=14) | beats seasonal naive at every h |",
        "|---|---|---|---|---|---|",
    ]
    pooled = b["pooled"]
    models = sorted(
        {r["model"] for r in pooled},
        key=lambda m: next(r["wape_pooled"] for r in pooled if r["model"] == m and r["horizon"] == 14),
    )
    for m in models:
        row = {r["horizon"]: r for r in pooled if r["model"] == m}
        beats = all(r["beats_snaive"] for r in row.values())
        name = f"**{m}**" if m == b["champion"] else m
        lines.append(
            f"| {name} | {row[7]['wape_pooled']:.1f}% | {row[14]['wape_pooled']:.1f}% | {row[28]['wape_pooled']:.1f}% | {row[14]['wape_vs_snaive']:.2f}x | {'yes' if beats else 'no'} |"
        )
    by_hotel = b["by_hotel"]
    lines += ["", "| Hotel | Champion WAPE h=7 | h=14 | h=28 | Seasonal naive h=14 |", "|---|---|---|---|---|"]
    for hotel in sorted({r["hotel"] for r in by_hotel}):
        champ = {r["horizon"]: r for r in by_hotel if r["hotel"] == hotel and r["model"] == b["champion"]}
        sn = next(r for r in by_hotel if r["hotel"] == hotel and r["model"] == "seasonal_naive" and r["horizon"] == 14)
        lines.append(
            f"| {hotel} | {champ[7]['wape_pooled']:.1f}% | {champ[14]['wape_pooled']:.1f}% | {champ[28]['wape_pooled']:.1f}% | {sn['wape_pooled']:.1f}% |"
        )
    lines += [
        "",
        f"{b['n_folds']} rolling-origin folds stepping {b['step_days']} days (origins {b['first_origin']} to {b['last_origin']}), every model refit per fold, "
        f"WAPE pooled over folds and both hotels; champion **{b['champion']}** = lowest WAPE at h=14 among models beating seasonal naive at every horizon. "
        f"Run {b['run_utc']} UTC in {b['seconds']} s; full table in `reports/backtest.md`.",
        "",
    ]
    return "\n".join(lines)


def elasticity_md() -> str:
    e = _load("elasticity.json")
    if not e:
        return "(no estimate yet: `make elasticity`)\n"
    n, c = e["naive"], e["controlled"]
    negative = [(k, r) for k, r in {**e["by_hotel"], **e["by_segment"]}.items() if r["ci_high"] < 0]
    lines = [
        f"The regression this data allows does **not** identify a demand elasticity: naive {n['elasticity']:+.2f} "
        f"(95% CI {n['ci_low']:.2f} to {n['ci_high']:.2f}), controlled {c['elasticity']:+.2f} "
        f"({c['ci_low']:.2f} to {c['ci_high']:.2f}; n = {c['n']} hotel-segment-weeks, R² {c['r2']:.2f}). "
        "Prices and bookings rise together because the hotels price to the demand they expect (simultaneity), so the "
        "coefficient recovers the pricing rule, not the demand curve.",
        "",
        "| Scope (controlled specification) | Elasticity | 95% CI |",
        "|---|---|---|",
    ]
    for h, r in e["by_hotel"].items():
        lines.append(f"| {h} | {r['elasticity']:+.2f} | {r['ci_low']:.2f} to {r['ci_high']:.2f} |")
    for seg, r in e["by_segment"].items():
        lines.append(f"| segment: {seg} | {r['elasticity']:+.2f} | {r['ci_low']:.2f} to {r['ci_high']:.2f} |")
    only = ", ".join(f"{k} ({r['elasticity']:+.2f})" for k, r in negative) or "none"
    sens = ", ".join(f"{v:.1f}" for v in e["sensitivity"])
    lines += [
        "",
        f"Clearly negative estimates: {only}. The pricing test and the price endpoint therefore run on a **planning assumption of "
        f"{e['planning_elasticity']:.1f}** ({e['planning_source']}), with sensitivity at {sens}; the response of the endpoint names the "
        "source so nobody mistakes it for a measurement. Details and the diagnosis: `reports/elasticity.md`, ADR 0002.",
        "",
    ]
    return "\n".join(lines)


def ab_md() -> str:
    a = _load("ab-test.json")
    if not a:
        return "(no A/B readout yet: `make ab`)\n"
    d, r, mc, db, mcb = (
        a["design"],
        a["result"],
        a["monte_carlo"],
        a["design_larger_change"],
        a["monte_carlo_larger_change"],
    )
    lo, hi = r["ci95_relative"]
    return "\n".join(
        [
            f"- Design: switchback, {d['days_per_arm']} days per arm, +{d['price_change']:.0%} price on B, metric revenue per day (mean {d['revenue_per_day_mean']:,.0f}, SD {d['revenue_per_day_sd']:,.0f}).",
            f"- Power: minimum detectable lift at 80% power **{d['mde_relative_at_target_power']:.1%}**; the elasticity ({d['elasticity']:.2f}) implies a lift of {d['expected_revenue_lift']:+.1%}, which this test detects with only **{d['power_at_expected_lift']:.0%}** power"
            + (
                f" (about {d['days_per_arm_needed_for_expected_lift']:.0f} days per arm would be needed)."
                if d["days_per_arm_needed_for_expected_lift"]
                else "."
            ),
            f"- Monte Carlo ({mc['simulations']} tests): empirical power {mc['empirical_power']:.0%}, mean estimated lift {mc['mean_estimated_lift']:+.1%} vs true {mc['true_lift']:+.1%}.",
            f"- Simulated test result: lift **{r['relative_lift']:+.1%}**, 95% bootstrap CI [{lo:+.1%}, {hi:+.1%}], Welch p = {r['p_value']:.3f}; decision: {'keep the higher price' if r['significant_at_5pct'] and r['relative_lift'] > 0 else 'no change on this evidence'}.",
            f"- A +{db['price_change']:.0%} change instead: expected lift {db['expected_revenue_lift']:+.1%}, analytic power {db['power_at_expected_lift']:.0%}, empirical {mcb['empirical_power']:.0%}.",
            "",
            "The full readout, with caveats and next steps, is `reports/ab-readout.md`.",
            "",
        ]
    )


def retrain_md() -> str:
    ok = _load("retrain-latest.json")
    blocked = _load("retrain-blocked-example.json")
    if not ok:
        return "(no retrain run yet: `make train`)\n"
    d = ok["drift"]
    lines = [
        f"- Latest run ({ok['run_utc']} UTC): {ok['model_type']} version {ok['version']}, holdout WAPE **{ok['holdout_wape']:.1f}%** "
        f"over the last {ok['holdout_days']} days; drift in the model's input columns {d['gate_drifted_share']:.0%} "
        f"(all columns {d['drifted_share']:.0%}, reference = the same weeks one year earlier); "
        f"**{'promoted' if ok['promoted'] else 'blocked'}** ({'; '.join(ok['reasons'])}).",
    ]
    if blocked:
        bd = blocked["drift"]
        lines.append(
            f"- Gate demonstration with a synthetic shock in the newest window (`dfp retrain --simulate-drift`): input drift "
            f"{bd['gate_drifted_share']:.0%}, version {blocked['version']} registered but **not promoted** ({'; '.join(blocked['reasons'])})."
        )
    lines.append("")
    return "\n".join(lines)


def replace(text: str, tag: str, body: str) -> str:
    start, end = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    a, b = text.index(start) + len(start), text.index(end)
    return text[:a] + "\n" + body + text[b:]


def main() -> None:
    blocks = {"BACKTEST": backtest_md(), "ELASTICITY": elasticity_md(), "AB": ab_md(), "RETRAIN": retrain_md()}
    if "--write" in sys.argv:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        for tag, body in blocks.items():
            text = replace(text, tag, body)
        readme.write_text(text, encoding="utf-8", newline="\n")
        print("wrote README.md")
    else:
        for tag, body in blocks.items():
            print(f"## {tag}\n{body}")


if __name__ == "__main__":
    main()
