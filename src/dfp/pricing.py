"""Price recommendation under constant elasticity. Revenue(p) = demand(p) * p with demand(p) = D0 * (p/p0)^e.
With |e| < 1 (inelastic) revenue keeps rising with price, so the recommendation is the top of the allowed
band and the response says so; with |e| > 1 there is an interior optimum only if a cost term exists, so the
search is over a bounded grid either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PricePoint:
    price: float
    change: float
    expected_demand: float
    expected_revenue: float


@dataclass
class Recommendation:
    base_price: float
    elasticity: float
    recommended_price: float
    recommended_change: float
    expected_demand: float
    expected_revenue: float
    revenue_change_vs_base: float
    inelastic: bool
    curve: list[PricePoint]


def recommend_price(
    base_price: float,
    expected_demand: float,
    elasticity: float,
    bounds: tuple[float, float] = (0.7, 1.3),
    steps: int = 61,
) -> Recommendation:
    if base_price <= 0 or expected_demand < 0:
        raise ValueError("base price must be positive and expected demand non-negative")
    changes = np.linspace(bounds[0] - 1.0, bounds[1] - 1.0, steps)
    curve: list[PricePoint] = []
    for c in changes:
        price = base_price * (1 + c)
        demand = expected_demand * (1 + c) ** elasticity
        curve.append(
            PricePoint(
                price=float(price),
                change=float(c),
                expected_demand=float(demand),
                expected_revenue=float(demand * price),
            )
        )
    best = max(curve, key=lambda p: p.expected_revenue)
    base_rev = expected_demand * base_price
    return Recommendation(
        base_price=base_price,
        elasticity=elasticity,
        recommended_price=best.price,
        recommended_change=best.change,
        expected_demand=best.expected_demand,
        expected_revenue=best.expected_revenue,
        revenue_change_vs_base=float(best.expected_revenue / base_rev - 1) if base_rev > 0 else 0.0,
        inelastic=abs(elasticity) < 1,
        curve=curve,
    )
