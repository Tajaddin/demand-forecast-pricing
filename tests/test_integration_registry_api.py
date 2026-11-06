"""Train, register, gate and serve against a throwaway sqlite MLflow store. Marked integration: a few
seconds of model fitting plus MLflow's registry."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from dfp.config import Settings, reset_settings_cache
from dfp.data import HOTELS
from dfp.retrain import holdout_wape, retrain
from dfp.tracking import production_version

pytestmark = pytest.mark.integration


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("DFP_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DFP_API_KEYS", "test-key")
    monkeypatch.setenv("DFP_RATE_LIMIT", "3/minute")
    monkeypatch.setenv("DFP_MODEL_NAME", "demand-forecast-test")
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()
    yield Settings()
    reset_settings_cache()


def test_first_train_promotes_then_the_gate_blocks_a_drifted_window(daily, env):
    first = retrain(daily, env, force_promote=True, model_type="seasonal_naive")
    assert first["promoted"] and first["version"] == "1"
    assert (env.reports_dir / "retrain-latest.json").exists()

    blocked = retrain(daily, env, simulate_drift=True, model_type="seasonal_naive")
    assert not blocked["promoted"]
    assert any("drift" in r for r in blocked["reasons"])
    assert blocked["drift"]["gate_drifted_share"] >= env.drift_share_threshold
    assert (env.reports_dir / "retrain-blocked-example.json").exists()

    worse = Settings(promotion_tolerance=-0.5)  # anything is "worse" than production minus 50%
    gated = retrain(daily, worse, model_type="seasonal_naive")
    assert not gated["promoted"]
    assert any("WAPE" in r for r in gated["reasons"])


def _harder_newest_weeks(daily, days):
    """Alternating +50% and -50% days in the newest window: every model forecasts it worse."""
    out = daily.copy()
    newest = out["arrival_date"] > out["arrival_date"].max() - np.timedelta64(days, "D")
    for hotel in HOTELS:
        idx = out.index[newest & (out["hotel"] == hotel)]
        out.loc[idx, "arrivals"] = out.loc[idx, "arrivals"] * np.where(np.arange(len(idx)) % 2 == 0, 1.5, 0.5)
    return out


def test_the_gate_scores_production_on_the_same_window(daily, env):
    drifted = retrain(daily, env, simulate_drift=True, model_type="seasonal_naive")
    assert not drifted["promoted"] and production_version(env) is None  # a version, but no alias yet

    first = retrain(daily, env, force_promote=True, model_type="seasonal_naive")
    hard = retrain(_harder_newest_weeks(daily, env.holdout_days), env, model_type="seasonal_naive")
    # the newest weeks are much harder than the window production was first scored on; that is not the
    # challenger's fault, and the same model type scored on the same window does no better
    assert hard["holdout_wape"] > first["holdout_wape"] * (1 + env.promotion_tolerance)
    assert not any("worse than production" in r for r in hard["reasons"]), hard["reasons"]
    assert hard["production_holdout_wape"] == pytest.approx(hard["holdout_wape"])

    challenger = retrain(daily, env, model_type="naive")
    assert challenger["production_model_type"] == "seasonal_naive"
    expected, _ = holdout_wape(daily, "seasonal_naive", env.holdout_days, None)
    assert challenger["production_holdout_wape"] == pytest.approx(expected)


def test_a_registry_read_error_blocks_the_run(daily, env, monkeypatch):
    retrain(daily, env, force_promote=True, model_type="seasonal_naive")
    failed = []

    def first_read_fails(original):
        def call(self, *args, **kwargs):
            if not failed:
                failed.append(original.__name__)
                raise MlflowException("registry temporarily unavailable", error_code="TEMPORARILY_UNAVAILABLE")
            return original(self, *args, **kwargs)

        return call

    with monkeypatch.context() as m:
        for name in ("get_registered_model", "get_model_version_by_alias"):
            m.setattr(MlflowClient, name, first_read_fails(getattr(MlflowClient, name)))
        with pytest.raises(MlflowException, match="temporarily unavailable"):
            retrain(daily, env, model_type="seasonal_naive")
    assert failed and production_version(env)["version"] == "1"


def test_api_serves_forecasts_prices_and_enforces_key_and_rate_limit(daily, env):
    retrain(daily, env, force_promote=True, model_type="lightgbm")
    (env.reports_dir).mkdir(parents=True, exist_ok=True)
    (env.reports_dir / "elasticity.json").write_text(
        json.dumps(
            {
                "identified": True,
                "planning_elasticity": -0.9,
                "planning_source": "controlled estimate",
                "by_hotel": {"City Hotel": {"elasticity": -0.7, "ci_high": -0.2}},
            }
        ),
        encoding="utf-8",
    )

    from dfp.serving.app import create_app

    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["model_loaded"]
        assert client.get("/model").json()["version"] == "1"
        assert client.get("/forecast", params={"hotel": "City Hotel"}).status_code == 401

        headers = {"X-API-Key": "test-key"}
        fc = client.get("/forecast", params={"hotel": "City Hotel", "horizon": 7}, headers=headers)
        assert fc.status_code == 200
        body = fc.json()
        assert len(body["forecast"]) == 7
        assert all(row["lo"] <= row["yhat"] <= row["hi"] for row in body["forecast"])
        assert client.get("/forecast", params={"hotel": "Nope"}, headers=headers).status_code == 404

        price = client.post("/price/recommend", json={"hotel": "City Hotel", "base_adr": 120}, headers=headers)
        assert price.status_code == 200
        p = price.json()
        assert (
            p["elasticity"] == -0.7
            and p["inelastic"]
            and len(p["curve"]) == 61
            and p["elasticity_source"].startswith("controlled")
        )
        assert p["recommended_change"] == pytest.approx(0.3)

        # 3/minute per key and route: the 404 above counted, so the third forecast call passes and the fourth
        # is refused; the price call is a different route with its own budget
        more = {"hotel": "Resort Hotel", "horizon": 3}
        assert client.get("/forecast", params=more, headers=headers).status_code == 200
        assert client.get("/forecast", params=more, headers=headers).status_code == 429
        metrics = client.get("/metrics").text
        assert "forecasts_served_total" in metrics and "rate_limit_rejections_total" in metrics
        reload = client.post("/model/reload", headers=headers)
        assert reload.status_code == 200 and reload.json()["loaded"]
