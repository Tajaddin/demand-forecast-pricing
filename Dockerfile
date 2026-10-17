# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

FROM base AS runtime
RUN groupadd -r app && useradd -r -g app -d /app app
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
# The reports the API reads (elasticity) and the data the retrain job needs; both are small.
COPY reports/elasticity.json ./reports/elasticity.json
COPY reports/backtest.json ./reports/backtest.json
COPY reports/backtest-residuals.csv ./reports/backtest-residuals.csv
COPY data/raw/hotels.csv ./data/raw/hotels.csv
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev && chown -R app:app /app
ENV PATH="/app/.venv/bin:$PATH" DFP_REPORTS_DIR=/app/reports
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 CMD curl -fsS http://127.0.0.1:8000/metrics >/dev/null || exit 1
CMD ["dfp", "serve", "--port", "8000"]
