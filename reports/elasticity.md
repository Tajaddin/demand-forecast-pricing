# Price elasticity: what the booking data says

Unit of analysis: hotel x market segment x ISO week with at least one non-cancelled priced booking.

| Specification | Elasticity | 95% CI | p | rows | R² |
|---|---|---|---|---|---|
| Naive (hotel, segment, month, year effects) | 0.08 | [-0.16, 0.32] | 0.521 | 1063 | 0.54 |
| Controlled (+ week of year, other hotel's demand, lead time) | 0.11 | [-0.14, 0.37] | 0.379 | 1063 | 0.59 |
| controlled, City Hotel | 0.98 | [0.11, 1.85] | 0.027 | 542 | 0.62 |
| controlled, Resort Hotel | 0.54 | [0.05, 1.03] | 0.032 | 521 | 0.64 |
| controlled, segment Corporate | -0.91 | [-1.31, -0.52] | 0.000 | 221 | 0.62 |
| controlled, segment Direct | -0.08 | [-0.60, 0.43] | 0.756 | 224 | 0.43 |
| controlled, segment Groups | 0.22 | [-0.81, 1.25] | 0.676 | 163 | 0.46 |
| controlled, segment Offline TA/TO | 0.22 | [-0.20, 0.64] | 0.304 | 228 | 0.56 |
| controlled, segment Online TA | 0.55 | [0.03, 1.06] | 0.037 | 227 | 0.67 |

**Diagnosis.** A positive or non-negative coefficient means prices and bookings rise together: the hotels set ADR with the demand they expect, so the regression recovers the pricing rule, not the demand curve (simultaneity). Controls for the demand the hotel could observe shrink the bias but cannot remove it without an instrument or an experiment.

**What the pricing work uses.** Elasticity -1.00 (assumption). Planning assumption, not an estimate from this data: unit elasticity (revenue flat to small price changes), bracketed by -0.5 and -1.5 in the sensitivity analysis. The A/B test exists to replace it with a measurement.
