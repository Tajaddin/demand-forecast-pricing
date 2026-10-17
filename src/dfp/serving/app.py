"""FastAPI service: forecasts from the production model in the MLflow registry and price recommendations."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from dfp import __version__
from dfp.config import Settings, settings
from dfp.data import HOTELS
from dfp.elasticity import load_elasticity
from dfp.logging_setup import configure_logging, get_logger
from dfp.pricing import recommend_price
from dfp.tracking import load_production_model, production_version

REQUESTS = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["method", "route", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
FORECASTS = Counter("forecasts_served_total", "Forecast rows served", ["hotel"])
RECOMMENDATIONS = Counter("price_recommendations_total", "Price recommendations served", ["hotel"])
MODEL_VERSION = Gauge("model_version", "Registry version of the loaded production model")
MODEL_LOADED = Gauge("model_loaded", "1 when a production model is loaded")
RATE_LIMITED = Counter("rate_limit_rejections_total", "Requests answered 429")


def _key_or_ip(request: Request) -> str:
    return request.headers.get("x-api-key") or get_remote_address(request)


class PriceRequest(BaseModel):
    hotel: str = Field(examples=list(HOTELS))
    base_adr: float = Field(gt=0, description="Current average daily rate")
    date: str | None = Field(
        default=None, description="YYYY-MM-DD within the forecast horizon; defaults to the next day"
    )
    expected_demand: float | None = Field(default=None, ge=0, description="Overrides the forecast")
    elasticity: float | None = Field(default=None, description="Overrides the estimate from the data")
    min_change: float = Field(default=-0.3, ge=-0.9, le=0)
    max_change: float = Field(default=0.3, ge=0, le=2)


log = get_logger("api")


class ModelState:
    model = None
    info: dict | None = None
    error: str | None = None

    @classmethod
    def load(cls, s: Settings) -> None:
        try:
            cls.model = load_production_model(s)
            cls.info = production_version(s)
            cls.error = None
            MODEL_LOADED.set(1)
            MODEL_VERSION.set(int(cls.info["version"]) if cls.info else 0)
            log.info("model loaded", version=cls.info["version"] if cls.info else None)
        except Exception as exc:
            cls.model, cls.info, cls.error = None, None, str(exc)
            MODEL_LOADED.set(0)
            log.warning("no production model loaded", error=str(exc)[:200])


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = settings()
    configure_logging(s.log_level, service="api")
    if s.otel_exporter_otlp_endpoint:
        _setup_tracing(app, s)
    ModelState.load(s)
    yield


def _setup_tracing(app: FastAPI, s: Settings) -> None:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "dfp-api"}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{s.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


def create_app() -> FastAPI:
    app = FastAPI(title="Demand forecast and pricing", version=__version__, lifespan=lifespan, docs_url="/docs")
    # One limiter per app: slowapi keys registered limits by endpoint name, so a limiter shared across
    # create_app() calls (the module-level app plus every test client) accumulates duplicate limits and
    # counts each request several times.
    limiter = Limiter(key_func=_key_or_ip, default_limits=[])
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, exc: RateLimitExceeded):
        RATE_LIMITED.inc()
        return JSONResponse(
            {"error": "rate limit exceeded", "detail": str(exc.detail)}, status_code=429, headers={"Retry-After": "60"}
        )

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response: Response = await call_next(request)
        route = request.scope.get("route")
        pattern = getattr(route, "path", request.url.path)
        REQUESTS.labels(request.method, pattern, str(response.status_code)).observe(time.perf_counter() - started)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            method=request.method,
            route=pattern,
            status=response.status_code,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        structlog.contextvars.unbind_contextvars("request_id")
        return response

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        keys = settings().api_key_set
        if keys and x_api_key not in keys:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    @app.get("/health")
    def health():
        ok = ModelState.model is not None
        body = {
            "status": "ok" if ok else "degraded",
            "model_loaded": ok,
            "model_version": ModelState.info["version"] if ModelState.info else None,
            "error": ModelState.error,
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model")
    def model_info():
        if not ModelState.info:
            raise HTTPException(status_code=503, detail=ModelState.error or "no production model")
        return {"name": settings().model_name, "alias": settings().model_alias, **ModelState.info}

    @app.post("/model/reload", dependencies=[Depends(require_key)])
    def model_reload():
        ModelState.load(settings())
        return {
            "loaded": ModelState.model is not None,
            "version": ModelState.info["version"] if ModelState.info else None,
            "error": ModelState.error,
        }

    @app.get("/forecast", dependencies=[Depends(require_key)])
    @limiter.limit(settings().rate_limit)
    def forecast(
        request: Request, hotel: str = Query(..., examples=list(HOTELS)), horizon: int = Query(14, ge=1, le=28)
    ):
        if ModelState.model is None:
            raise HTTPException(status_code=503, detail=ModelState.error or "no production model")
        if hotel not in HOTELS:
            raise HTTPException(status_code=404, detail=f"unknown hotel; one of {list(HOTELS)}")
        out = ModelState.model.predict(pd.DataFrame({"hotel": [hotel], "horizon": [horizon]}))
        FORECASTS.labels(hotel).inc(len(out))
        rows = out.drop(columns=["hotel"]).to_dict(orient="records")
        return {
            "hotel": hotel,
            "horizon": horizon,
            "model_version": ModelState.info["version"] if ModelState.info else None,
            "forecast": rows,
        }

    @app.post("/price/recommend", dependencies=[Depends(require_key)])
    @limiter.limit(settings().rate_limit)
    def price(request: Request, body: PriceRequest):
        if body.hotel not in HOTELS:
            raise HTTPException(status_code=404, detail=f"unknown hotel; one of {list(HOTELS)}")
        s = settings()
        if body.elasticity is not None:
            elasticity, source = body.elasticity, "request"
        else:
            elasticity, source = load_elasticity(Path(s.reports_dir) / "elasticity.json", body.hotel)
        demand = body.expected_demand
        date_used = body.date
        if demand is None:
            if ModelState.model is None:
                raise HTTPException(status_code=503, detail="no production model; pass expected_demand")
            fc = ModelState.model.predict(pd.DataFrame({"hotel": [body.hotel], "horizon": [28]}))
            row = fc[fc["date"] == body.date] if body.date else fc.iloc[[0]]
            if row.empty:
                raise HTTPException(status_code=400, detail="date must be within the next 28 days")
            demand = float(row.iloc[0]["yhat"])
            date_used = str(row.iloc[0]["date"])
        rec = recommend_price(body.base_adr, demand, elasticity, bounds=(1 + body.min_change, 1 + body.max_change))
        RECOMMENDATIONS.labels(body.hotel).inc()
        return {
            "hotel": body.hotel,
            "date": date_used,
            "elasticity": elasticity,
            "elasticity_source": source,
            "expected_demand_at_base": demand,
            "recommended_price": round(rec.recommended_price, 2),
            "recommended_change": round(rec.recommended_change, 3),
            "expected_revenue_change": round(rec.revenue_change_vs_base, 4),
            "inelastic": rec.inelastic,
            "note": "demand is inelastic in this range, so revenue rises with price up to the allowed maximum"
            if rec.inelastic
            else "interior optimum within the allowed range",
            "curve": [
                {
                    "change": round(p.change, 3),
                    "price": round(p.price, 2),
                    "demand": round(p.expected_demand, 2),
                    "revenue": round(p.expected_revenue, 2),
                }
                for p in rec.curve
            ],
        }

    return app


app = create_app()
