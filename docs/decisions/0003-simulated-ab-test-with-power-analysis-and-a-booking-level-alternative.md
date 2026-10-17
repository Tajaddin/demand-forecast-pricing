# 0003. A simulated switchback A/B test with a power analysis, and the booking-level design it argues for

Date: 2026-10-17. Status: accepted.

## Context

There is no way to run a real pricing experiment on a public dataset. What can be shown is the discipline:
design, power, analysis and a readout a business team can act on, with the truth known so the analysis can
be checked.

## Decision

- **Design under test**: an 8-week switchback on one hotel, each day randomly A or B, B priced +5%, metric
  revenue per day. Outcomes are simulated from the hotel's own last 8 weeks (weekday profile, day-to-day
  noise) and the planning elasticity, with fixed seeds.
- **Power** is computed analytically (two-sample t on revenue per day) and checked by Monte Carlo (2,000
  simulated tests). The honest result is that a day-level test is badly underpowered for realistic lifts:
  the minimum detectable lift at 80% power is around 18%.
- **Sensitivity**: expected lift and power at elasticities -0.5, -1.0 and -1.5.
- **The better design** is reported with numbers: randomising at the booking (quote) level turns ~28 days
  per arm into thousands of bookings per arm and brings the minimum detectable lift down by an order of
  magnitude.
- **Analysis**: Welch t-test, 10,000-sample bootstrap CI for the relative lift, a plain-language decision,
  caveats (switchback carry-over through booking lead times, cancellations, fat tails), next steps.

## Consequences

- The readout's conclusion is "the switchback cannot answer this; run a booking-level test for N bookings",
  which is the correct advice and more useful than a flattering simulated win.
- Everything is reproducible from `dfp ab` with seeds; `reports/ab-test.json` holds every number in the
  readout.
