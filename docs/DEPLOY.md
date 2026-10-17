# Running and deploying

## Local

```bash
uv sync --extra dev
cp .env.example .env
docker compose up -d mlflow            # registry on http://localhost:5002
export MLFLOW_TRACKING_URI=http://localhost:5002
make data elasticity ab                # reports/data-quality.json, elasticity.*, ab-readout.md
make backtest                          # reports/backtest.md (a few minutes)
make train                             # registers the champion, alias production
make retrain-blocked                   # demonstrates the gate
make serve                             # http://localhost:8100/docs
```

The API in a container: `make up` builds the image and starts it on :8100 against the compose MLflow.
`make up-obs` adds Prometheus (:9103), Alertmanager (:9104) and Grafana (:3006, dashboard "Demand forecast API").

```bash
curl "http://localhost:8100/forecast?hotel=City%20Hotel&horizon=7" -H "X-API-Key: $DFP_API_KEYS"
curl -X POST http://localhost:8100/price/recommend -H "Content-Type: application/json" -H "X-API-Key: $DFP_API_KEYS" \
  -d '{"hotel":"City Hotel","base_adr":120}'
```

## Tests and checks

| Command | What |
|---|---|
| `make lint`, `make typecheck` | ruff (lint and format), mypy |
| `make test-unit` | data pipeline, metrics, models, backtest, elasticity, A/B, pricing, drift parser |
| `make test-integration` | train, gate (drift and accuracy), API with key and rate limit, against a sqlite MLflow |
| `make alerts-test` | promtool tests for the alert rules |
| `make tf-validate` | Terraform fmt and validate in a container |
| `make image` | the API image |

## Scheduled retrain

`.github/workflows/retrain.yml` runs `dfp retrain` every Monday against `MLFLOW_TRACKING_URI` from the
`production` environment secrets and keeps the reports as an artifact. The Compose `jobs` profile runs the same
command: `docker compose --profile jobs run --rm retrain`.

## Fly.io (free allowance)

```bash
export MLFLOW_USER=admin MLFLOW_PASSWORD='...' DFP_API_KEYS='...'
make deploy
```

`scripts/deploy.sh` creates two apps: `dfp-mlflow` (MLflow server with a 1 GB volume, basic auth; create the
first user with `fly ssh console -a dfp-mlflow -C "mlflow auth create-user ..."` as the MLflow docs describe) and
`dfp-api`, with `MLFLOW_TRACKING_URI` and `DFP_API_KEYS` as secrets. Then populate the registry from this machine:

```bash
MLFLOW_TRACKING_URI=https://admin:...@dfp-mlflow.fly.dev make backtest train
curl -X POST https://dfp-api.fly.dev/model/reload -H "X-API-Key: ..."
```

## AWS App Runner (Terraform)

```bash
cd infra/terraform && cp terraform.tfvars.example terraform.tfvars   # MLflow URL, API keys
terraform init && terraform apply                                     # ECR repo, App Runner service, SSM secrets
aws ecr get-login-password | docker login --username AWS --password-stdin "$(terraform output -raw ecr_repository_url | cut -d/ -f1)"
docker build -t "$(terraform output -raw ecr_repository_url):0.1.0" .. && docker push "$(terraform output -raw ecr_repository_url):0.1.0"
```

App Runner redeploys on push (auto deployments). Smallest size, about $5 a month while it runs.

## Teardown

`make teardown` destroys the Fly apps and the Terraform resources; `make down` removes the local containers and
the MLflow volume.
