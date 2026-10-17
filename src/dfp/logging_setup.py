from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", service: str = "dfp") -> None:
    """JSON lines on stdout, one event per line, with the service name on every record."""
    logging.basicConfig(level=level.upper(), stream=sys.stdout, format="%(message)s")
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "mlflow", "urllib3", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str = "dfp") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
