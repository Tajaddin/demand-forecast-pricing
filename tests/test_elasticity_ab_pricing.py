import json

import numpy as np
import pandas as pd
import pytest

from dfp.ab import (
    analyze,
    baseline_from_history,
    expected_lift,
    monte_carlo_power,
    power_analysis,
    simulate_test,
    write_readout,
)
from dfp.data import weekly_segment_panel
from dfp.drift import _drifted_share, drift_frame, run_drift
from dfp.elasticity import estimate, implied_demand_multiplier, load_elasticity, write_report
from dfp.pricing import recommend_price


def test_elasticity_recovers_the_synthetic_slope(bookings, tmp_path):
    # the generator uses demand ~ price^-0.8 with noisy prices; the weekly panel should land near -0.8
    result = estimate(weekly_segment_panel(bookings))
    # weekly, booking-weighted mean prices attenuate the daily slope of -0.8 toward zero; the sign and the
    # significance are what the estimator must get right when prices really are exogenous
    assert result["naive"]["elasticity"] < 0
    c = result["controlled"]
    assert -2.0 < c["elasticity"] < -0.3, c
    assert c["ci_high"] < 0, c
    assert set(result["by_hotel"]) == {"City Hotel", "Resort Hotel"}
    assert result["identified"] and result["planning_source"] == "controlled estimate"
    jp, mp = write_report(result, tmp_path)
    value, source = load_elasticity(jp, "City Hotel")
    assert source.startswith("controlled") and value < 0
    assert load_elasticity(tmp_path / "missing.json", default=-1.5) == (
        -1.5,
        "default assumption (no elasticity report)",
    )
    assert "Diagnosis" in mp.read_text(encoding="utf-8")


def test_implied_demand_and_expected_lift():
    assert implied_demand_multiplier(0.10, -1.0) == pytest.approx(1 / 1.1)
    assert expected_lift(0.10, -1.0) == pytest.approx(0.0)
    assert expected_lift(0.10, -0.5) > 0
    assert expected_lift(0.10, -2.0) < 0


def test_price_recommendation_follows_elasticity():
    inelastic = recommend_price(100, 50, -0.5)
    assert inelastic.inelastic and inelastic.recommended_change == pytest.approx(0.3)
    elastic = recommend_price(100, 50, -2.5)
    assert not elastic.inelastic and elastic.recommended_change == pytest.approx(-0.3)
    unit = recommend_price(100, 50, -1.0)
    assert abs(unit.revenue_change_vs_base) < 1e-9
    assert len(unit.curve) == 61
    with pytest.raises(ValueError, match="positive"):
        recommend_price(0, 10, -1)


def test_ab_design_simulation_and_analysis_are_coherent(daily, bookings, tmp_path):
    base = baseline_from_history(daily, bookings, "City Hotel")
    assert set(base.weekday_demand) == set(range(7))
    design = power_analysis(base, days_per_arm=28, delta=0.05, elasticity=-0.8)
    assert 0 < design["mde_relative_at_target_power"] < 1
    assert 0 < design["power_at_expected_lift"] <= 1
    bigger = power_analysis(base, days_per_arm=28, delta=0.3, elasticity=-0.8)
    assert bigger["power_at_expected_lift"] > design["power_at_expected_lift"]

    test = simulate_test(base, delta=0.05, elasticity=-0.8, days=56, seed=1)
    assert test["arm"].value_counts().to_dict() == {"A": 28, "B": 28}
    assert (test.loc[test["arm"] == "B", "adr"].mean() / test.loc[test["arm"] == "A", "adr"].mean()) == pytest.approx(
        1.05, rel=0.1
    )
    result = analyze(test, n_boot=500, seed=1)
    assert result["ci95_relative"][0] < result["relative_lift"] < result["ci95_relative"][1]
    assert 0 <= result["p_value"] <= 1

    # a large price change with a weak demand response is a big revenue lift and easy to detect
    mc = monte_carlo_power(base, delta=0.5, elasticity=-0.2, days=56, n_sims=40, seed=3)
    assert mc["empirical_power"] > 0.8
    from dfp.ab import booking_level_power

    bl = booking_level_power(bookings, "City Hotel", 56, 0.05, -0.8)
    assert bl["bookings_per_arm"] > 100 and bl["mde_relative_at_target_power"] < design["mde_relative_at_target_power"]
    jp, mp = write_readout(base, design, bigger, result, mc, mc, tmp_path, 0.05, 0.3, [design], bl, "assumption")
    text = mp.read_text(encoding="utf-8")
    assert "Decision" in text and "Caveats" in text and "Sensitivity" in text
    assert json.loads(jp.read_text(encoding="utf-8"))["design"]["days_per_arm"] == 28


def test_drift_parser_and_report(daily, tmp_path):
    ref = drift_frame(daily)
    cur = ref.sample(200, random_state=0)  # the same distribution: little or no drift
    same = run_drift(ref, cur, tmp_path, name="same")
    cur_shifted = cur.copy()
    cur_shifted["arrivals"] *= 3
    cur_shifted["adr_mean"] *= 2
    shifted = run_drift(ref, cur_shifted, tmp_path, name="shifted")
    assert same["drifted_share"] < 0.5
    assert shifted["drifted_share"] > same["drifted_share"]
    assert shifted["per_column"]["arrivals"]["drifted"] and shifted["per_column"]["adr_mean"]["drifted"]
    assert not same["per_column"]["arrivals"]["drifted"]
    assert (tmp_path / "shifted.html").exists()
    share, count, per = _drifted_share(
        {
            "metrics": [
                {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 2, "share": 0.5}},
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "a",
                        "method": "K-S p_value",
                        "threshold": 0.05,
                    },
                    "value": 0.01,
                },
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "b",
                        "method": "Wasserstein",
                        "threshold": 0.1,
                    },
                    "value": 0.05,
                },
            ]
        }
    )
    assert (share, count) == (0.5, 2)
    assert per["a"]["drifted"] and not per["b"]["drifted"]
    assert isinstance(np.float64(share), float) and isinstance(pd.DataFrame(), pd.DataFrame)
