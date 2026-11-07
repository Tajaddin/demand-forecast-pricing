"""Price recommendation under constant elasticity. Demand(p) = D0 * (p/p0)^e, rooms sold are capped at capacity
when one is given, and the objective is (p - unit_cost) * rooms sold, which is revenue when there is no cost.
With |e| < 1 (inelastic) revenue keeps rising with price, so the recommendation is the top of the allowed band
and the response says so; with |e| > 1 there is an interior optimum only if a cost or capacity term exists, and
at e = -1 revenue is flat. The search is over a bounded grid that contains the current price; prices that tie
for the best objective resolve to the smallest change, so a flat curve recommends no change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Profits within this relative distance of the best count as a tie: rounding noise is about 1e-16.
TIE_TOLERANCE = 1e-9


@dataclass
class PricePoint:
    price: float
    change: float
    expected_demand: float  # rooms sold: demand capped at capacity
    expected_revenue: float
    expected_profit: float


@dataclass
class Recommendation:
    base_price: float
    elasticity: float
    recommended_price: float
    recommended_change: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    revenue_change_vs_base: float
    inelastic: bool
    flat: bool  # every price in the band ties for the best profit
    curve: list[PricePoint]


def recommend_price(
    base_price: float,
    expected_demand: float,
    elasticity: float,
    bounds: tuple[float, float] = (0.7, 1.3),
    steps: int = 61,
    capacity: float | None = None,
    unit_cost: float = 0.0,
) -> Recommendation:
    if base_price <= 0 or expected_demand < 0:
        raise ValueError("base price must be positive and expected demand non-negative")
    if capacity is not None and capacity <= 0:
        raise ValueError("capacity must be positive")
    if unit_cost < 0:
        raise ValueError("unit cost must be non-negative")
    changes = np.linspace(bounds[0] - 1.0, bounds[1] - 1.0, steps)
    changes[np.isclose(changes, 0.0, rtol=0.0, atol=1e-12)] = 0.0
    if changes[0] <= 0.0 <= changes[-1] and 0.0 not in changes:
        changes = np.sort(np.append(changes, 0.0))
    curve: list[PricePoint] = []
    for c in changes:
        price = float(base_price * (1 + c))
        sold = float(expected_demand * (1 + c) ** elasticity)
        if capacity is not None:
            sold = min(sold, capacity)
        curve.append(
            PricePoint(
                price=price,
                change=float(c),
                expected_demand=sold,
                expected_revenue=sold * price,
                expected_profit=sold * (price - unit_cost),
            )
        )
    top = max(p.expected_profit for p in curve)
    ties = [p for p in curve if p.expected_profit >= top - TIE_TOLERANCE * max(abs(top), 1.0)]
    best = min(ties, key=lambda p: abs(p.change))
    base_sold = expected_demand if capacity is None else min(expected_demand, capacity)
    base_rev = base_sold * base_price
    return Recommendation(
        base_price=base_price,
        elasticity=elasticity,
        recommended_price=best.price,
        recommended_change=best.change,
        expected_demand=best.expected_demand,
        expected_revenue=best.expected_revenue,
        expected_profit=best.expected_profit,
        revenue_change_vs_base=float(best.expected_revenue / base_rev - 1) if base_rev > 0 else 0.0,
        inelastic=abs(elasticity) < 1,
        flat=len(ties) == len(curve),
        curve=curve,
    )
