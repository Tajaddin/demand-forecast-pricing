"""Train, register, gate and serve against a throwaway sqlite MLflow store. Marked integration: a few
seconds of model fitting plus MLflow's registry."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from dfp.config import Settings, reset_settings_cache
from dfp.retrain import retrain

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
