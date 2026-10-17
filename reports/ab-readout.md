# A/B pricing test readout: City Hotel, +5% on treatment days

**Simulated test.** Outcomes come from the hotel's own recent seasonality, the elasticity estimated from the
data and its day-to-day noise, so the true effect is known; this is a rehearsal of the design and the analysis,
not an experiment that was run with guests.

## Question

Does raising prices by 5% increase revenue per day at the City Hotel?

## Design

- Switchback: each of 56 days is randomly A (current prices) or B (+5%); 28 days per arm.
- Metric: revenue per day. Over the last 56 days it averaged 28,314 with a day-to-day SD of 6,657.
- Elasticity used for planning: -1.00 (assumption); under it a +5% price moves revenue by +0.0%.
- Power: with 28 days per arm the smallest lift detectable at 80% power is 17.9%; the expected lift gives only 5% power.
- Monte Carlo check (2000 simulated tests): empirical power 5%, mean estimated lift +0.2% against a true +0.0%.

## Result

- Revenue per day: A 24,971, B 27,087; difference +2,116 (+8.5%).
- 95% bootstrap CI for the relative lift: [-2.8%, +19.2%]; Welch t = 1.48, p = 0.145.
- Arrivals per day: A 63.3, B 65.1.

## Decision

Do not change prices on this evidence; the test could not tell the lift from zero.

## Sensitivity to the elasticity assumption

| Elasticity | Expected lift from +5% | Power with 28 days per arm |
|---|---|---|
| -0.50 | +2.5% | 7% |
| -1.00 | +0.0% | 5% |
| -1.50 | -2.4% | 7% |

## A design that can actually answer the question

Randomising bookings instead of days and measuring revenue per booking gives about 1,800 bookings per arm in the same 8 weeks (mean 343, SD 235): the minimum detectable lift at 80% power drops to **6.4%**, and the +5% change under the planning elasticity would be detected with 5% power. It needs price randomisation at quote time (per booking request), which the switchback does not.

## What a larger change would look like

With +15% instead, the expected lift is +0.0% and the same 8 weeks give 5% analytic power (5% empirical over 2000 simulations).

## Caveats

- The elasticity is a planning assumption: the observational estimate from this data is not identified (prices and bookings rise together because the hotels price to expected demand; see reports/elasticity.md).
- Switchback days are not independent of each other: bookings are made ahead of the stay, so a price seen on day t influences demand on later days. A real test should randomise by booking date or by comparable units.
- Revenue per day ignores cancellations after booking and ancillary spend.
- Simulated noise is normal around a weekday profile; real demand has fatter tails (events, holidays).

## Next steps

1. Run the real test for the number of days the power analysis asks for, randomising by booking date.
2. Pre-register the metric, the stopping rule and the analysis (this document).
3. Re-estimate elasticity from the experiment and feed it to the price recommendation endpoint.
