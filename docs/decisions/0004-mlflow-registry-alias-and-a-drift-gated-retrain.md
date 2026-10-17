# 0004. MLflow registry with a production alias, and a retrain job that drift or accuracy can block

Date: 2026-10-17. Status: accepted.

## Context

The spec asks for tracking and a registry, a serving API, and a scheduled retrain whose promotion a drift
check can block. The failure mode to avoid is a cron job that silently replaces a good model with a worse one
because the data changed.

## Decision

- **MLflow 3** with a database-backed store (sqlite locally and in CI, the compose server for the API), one
  run per backtest and per (re)train, the packaged model as a `pyfunc` (per-hotel forecasters plus residual
  quantiles), registered as `demand-forecast`; serving loads `models:/demand-forecast@production`.
- **Retrain** (`dfp retrain`): refit the champion type on the newest window, score the last 28 days as a
  holdout, run Evidently's data-drift preset with the newest 28 days as current and, as reference, the same
  calendar weeks one year earlier widened by two weeks (comparing the newest weeks with a whole year flagged
  ordinary seasonality as drift in the first version). The gate judges only the columns the forecaster
  consumes (arrivals, room-nights, cancellations, weekday); drift in prices, revenue and lead time is reported
  for a human but does not block, because the arrivals model does not read them. **Promote only if** the
  input-column drift share < 30% **and** holdout WAPE is not worse than production by more than 5%
  relative. Every run registers a version and writes `reports/retrain-latest.json` with the decision and its
  reasons; a blocked run leaves the alias alone.
- `dfp retrain --simulate-drift` injects a synthetic shock into the newest window so the blocked path is
  exercised and recorded (`reports/retrain-blocked-example.json`); the integration test covers both paths.
- Scheduled weekly by GitHub Actions against the production MLflow URL; a Compose `jobs` profile runs the same
  command the way a cron container would.

## Consequences

- Promotion is explainable: the JSON says which gate failed and by how much.
- The holdout is the last 28 days of a static dataset, so locally "retrain" is a rehearsal; with live data
  the same code compares genuinely new weeks.
- On this data the normal retrain passes (0% input drift, 38% across all columns because 2017 prices and
  lead times differ from 2016) and the synthetic shock is blocked (50% input drift), which is the behaviour
  a gate should have.
- Evidently's report is saved as HTML next to the JSON for a human to look at.
