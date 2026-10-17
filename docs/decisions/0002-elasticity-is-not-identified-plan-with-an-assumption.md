# 0002. The observational elasticity is not identified; the pricing work plans with a stated assumption

Date: 2026-10-17. Status: accepted.

## Context

The spec asks for a price-elasticity estimate. A regression of log weekly bookings on log ADR with hotel,
segment, month and year effects returns a **positive** coefficient (+0.08 overall, +0.98 for the City Hotel):
higher prices go with more bookings. That is simultaneity, not a demand curve: the hotels set ADR with the
demand they expect, so price is a function of demand in this data.

## Decision

- Report the naive estimate and a **controlled** one (week-of-year effects, the other hotel's bookings the
  same week as a market-demand proxy, mean lead time). The controlled estimate stays positive and
  insignificant overall; only the Corporate segment, whose rates are negotiated rather than yield-managed,
  shows a clearly negative elasticity (about -0.9). The report says all of this.
- Do **not** feed a positive "elasticity" into a pricing simulation. The A/B design and the price
  recommendation endpoint use a **planning assumption** of -1.0 (unit elastic: revenue flat to small price
  changes), with -0.5 and -1.5 as the sensitivity range, labelled as an assumption everywhere it appears
  (`elasticity_source` in the API response, the readout, the README).
- The endpoint uses a measured value only when a controlled estimate's confidence interval lies below zero
  (`identified`), which this data does not provide.

## Consequences

- The project does not claim a demand elasticity it cannot support, and explains why the experiment is the
  right next step; that is the point a pricing team would want made.
- The pricing recommendation is only as good as the assumption; the response carries the source so a
  caller cannot mistake it for a measurement.
- An instrument (for example cost shocks or competitor prices) or booking-level price randomisation would
  identify the elasticity; neither exists in this dataset.
