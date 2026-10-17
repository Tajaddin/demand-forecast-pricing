from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every value can be set as DFP_<NAME>; MLflow's own variable is honoured too."""

    model_config = SettingsConfigDict(env_prefix="DFP_", env_file=".env", extra="ignore")

    data_path: Path = Path("data/raw/hotels.csv")
    reports_dir: Path = Path("reports")
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI", "DFP_MLFLOW_TRACKING_URI"),
    )
    mlflow_experiment: str = "demand-forecast"
    model_name: str = "demand-forecast"
    model_alias: str = "production"

    # serving
    api_keys: str = ""  # comma-separated; empty = open (local development only)
    rate_limit: str = "60/minute"
    log_level: str = "INFO"
    otel_exporter_otlp_endpoint: str = Field(
        default="", validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT", "DFP_OTEL_EXPORTER_OTLP_ENDPOINT")
    )

    # backtest
    backtest_folds: int = 12
    backtest_step_days: int = 14
    horizons: tuple[int, ...] = (7, 14, 28)
    season_length: int = 7

    # retrain gate
    holdout_days: int = 28
    drift_share_threshold: float = 0.3
    promotion_tolerance: float = 0.05

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    settings.cache_clear()
