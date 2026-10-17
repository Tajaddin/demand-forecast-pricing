# 0001. Baselines first, and a rolling-origin backtest that refits every model per fold

Date: 2026-10-17. Status: accepted.

## Context

Forecast accuracy numbers are easy to inflate: a single train/test split, a model tuned on the test window, or
a comparison without a naive baseline. The README has to carry one number a reviewer can trust.

## Decision

- Every model is scored against **naive** (last value) and **seasonal naive** (same weekday last week). A
  model that does not beat seasonal naive at every horizon is marked as such and cannot be the champion; if
  nothing beats it, seasonal naive is the champion and the README says so.
- **Rolling origin**: 12 folds stepping 14 days over the last ~6 months, horizons 7, 14 and 28 days, every
  model refit on all data before each origin. Nothing is tuned on the test windows; LightGBM's parameters are
  fixed in code.
- Metrics: **WAPE** (total absolute error over total arrivals) is the headline because it is robust to zero
  days and pools naturally; MAPE, sMAPE and MASE are reported alongside.
- Champion = lowest pooled WAPE at h=14 among models that beat seasonal naive at every horizon.
- Per-day residuals from the backtest become the model's prediction intervals (10th and 90th percentiles per
  horizon step), so the intervals reflect out-of-sample error, not in-sample fit.

## Consequences

- The backtest takes a few minutes (two hotels x 12 folds x 6 models); it runs locally and a shorter version
  runs in CI.
- Prophet is included but optional at import, so an environment without a working CmdStan still runs.
- The LightGBM model is one direct multi-horizon model (origin, step) rather than 28 models, which keeps the
  fold loop fast at the cost of sharing parameters across steps.
