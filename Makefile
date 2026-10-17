.PHONY: install data backtest elasticity ab train retrain retrain-blocked serve up up-obs down lint format typecheck test test-unit test-integration alerts-test image tf-validate deploy teardown report

install:
	uv sync --extra dev

data:           ## download the dataset, build the series, write the quality report
	uv run dfp data

backtest:       ## rolling-origin backtest of every model; reports/backtest.md and MLflow
	uv run dfp backtest

elasticity:
	uv run dfp elasticity

ab:             ## design, simulate and read out the pricing test
	uv run dfp ab

train:          ## register the champion and point the production alias at it
	uv run dfp train

retrain:        ## the scheduled job: gated promotion
	uv run dfp retrain

retrain-blocked: ## demonstrates the gate with a synthetic shock in the newest window
	uv run dfp retrain --simulate-drift

serve:
	uv run dfp serve --port 8100

up:             ## MLflow server (registry) and the API from the local image
	docker compose up -d --build mlflow api

up-obs:
	docker compose --profile observability up -d

down:
	docker compose --profile observability --profile jobs down -v

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=dfp --cov-report=term-missing --cov-report=xml

test-unit:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

alerts-test:
	docker run --rm -v "$(CURDIR)/observability:/w" -w /w --entrypoint promtool prom/prometheus:v3.5.0 test rules alerts_test.yml

image:
	docker build -t dfp/api:local .

tf-validate:
	docker run --rm -v "$(CURDIR)/infra/terraform:/w" -w /w hashicorp/terraform:1.9 init -backend=false -input=false >/dev/null && docker run --rm -v "$(CURDIR)/infra/terraform:/w" -w /w hashicorp/terraform:1.9 validate

deploy:         ## Fly.io (see docs/DEPLOY.md); Terraform for App Runner lives in infra/terraform
	bash scripts/deploy.sh

teardown:
	bash scripts/teardown.sh

report:         ## render reports/*.json into the README between the markers
	uv run python scripts/render_results.py --write
