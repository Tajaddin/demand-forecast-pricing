# 0005. Serving with keys and rate limits, Prometheus metrics with tested alerts, and two deployment paths

Date: 2026-10-17. Status: accepted.

## Context

The API is the part a product team integrates with; it has to be safe to expose, observable, and cheap to run.

## Decision

- **FastAPI** with `X-API-Key` on `/forecast`, `/price/recommend` and `/model/reload` (open only when no keys
  are configured, for local work), a per-key rate limit (slowapi), request ids, structured JSON logs
  (structlog), and `prom-client` metrics: request latency by route pattern, forecast rows and recommendations
  served, model version and `model_loaded`, rate-limit rejections.
- The model is loaded at start-up from the registry alias and can be reloaded without a restart; when there
  is no production model the service stays up, `/health` returns 503 and the `NoProductionModel` alert fires.
- **Alerts**: ApiDown, NoProductionModel, HighErrorRate, SlowForecasts, RateLimitSpike, each with promtool
  tests; a Grafana dashboard in Compose. OpenTelemetry traces are opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT`.
- **Deployment**: Fly.io for the free path (`fly.toml` for the API, `fly.mlflow.toml` for an MLflow server
  with a volume and basic auth) and **Terraform for AWS App Runner + ECR** as the infrastructure-as-code path
  (validated in CI, not applied; about $5/month at the smallest size). Secrets come from Fly secrets or SSM
  parameters, never from the image.
- One Dockerfile built with uv, non-root, with the small report files the API reads baked in.

## Consequences

- The API has no database of its own: MLflow is its only dependency, which keeps the deployment to one
  container plus a tracking server.
- Rate limiting is per process; across several instances a shared store would be needed.
- The image measures 3.7 GB because serving loads the same environment that trains (Prophet with CmdStan,
  LightGBM, MLflow, Evidently, statsmodels); a slimmer serving image would drop the training-only
  dependencies, which the registry's pip requirements would then have to match.
