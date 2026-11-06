"""The API with a registry it cannot reach: what an unauthenticated caller learns, and when keys are required."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dfp.config import reset_settings_cache

SECRET = "NOT-A-REAL-SECRET"
TRACKING_URI = f"http://audituser:{SECRET}@127.0.0.1:9"


@pytest.fixture
def no_registry(tmp_path, monkeypatch):
    """MLflow fails the way an unreachable server does: the exception text carries the tracking URI."""

    def unreachable(s):
        raise ConnectionError(f"API request to {s.mlflow_tracking_uri}/api/2.0/mlflow/experiments/get-by-name failed")

    monkeypatch.chdir(tmp_path)  # no developer .env
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    monkeypatch.setattr("dfp.serving.app.load_production_model", unreachable)
    reset_settings_cache()
    yield
    reset_settings_cache()


def _client():
    from dfp.serving.app import create_app

    return TestClient(create_app())


def test_registry_errors_stay_out_of_responses(no_registry, monkeypatch):
    monkeypatch.setenv("DFP_API_KEYS", "test-key")
    reset_settings_cache()
    with _client() as client:
        headers = {"X-API-Key": "test-key"}
        responses = [
            client.get("/health"),
            client.get("/model"),
            client.get("/forecast", params={"hotel": "City Hotel"}, headers=headers),
            client.post("/model/reload", headers=headers),
        ]
    assert [r.status_code for r in responses] == [503, 503, 503, 200]
    assert responses[0].json()["model_loaded"] is False
    for r in responses:
        assert SECRET not in r.text and "127.0.0.1" not in r.text, r.text


def test_log_lines_drop_credentials_from_urls():
    from dfp.logging_setup import redact_url_credentials

    event = {"event": "no production model loaded", "error": f"API request to {TRACKING_URI}/api/2.0 failed", "n": 3}
    out = redact_url_credentials(None, "warning", event)
    assert out["error"] == "API request to http://***@127.0.0.1:9/api/2.0 failed"
    assert out["n"] == 3


def test_keyed_routes_refuse_requests_when_no_keys_are_configured(no_registry, monkeypatch):
    monkeypatch.setenv("DFP_API_KEYS", "")
    reset_settings_cache()
    with _client() as client:
        assert client.post("/model/reload").status_code == 401
        assert client.post("/model/reload", headers={"X-API-Key": ""}).status_code == 401
        assert client.get("/forecast", params={"hotel": "City Hotel"}).status_code == 401
        price = client.post("/price/recommend", json={"hotel": "City Hotel", "base_adr": 100, "expected_demand": 10})
        assert price.status_code == 401
        assert client.get("/health").status_code == 503  # health and metrics stay open

    monkeypatch.setenv("DFP_ALLOW_UNAUTHENTICATED", "true")
    reset_settings_cache()
    with _client() as client:
        assert client.post("/model/reload").status_code == 200


def test_the_development_flag_does_not_open_a_keyed_service(no_registry, monkeypatch):
    monkeypatch.setenv("DFP_API_KEYS", "test-key")
    monkeypatch.setenv("DFP_ALLOW_UNAUTHENTICATED", "true")
    reset_settings_cache()
    with _client() as client:
        assert client.post("/model/reload").status_code == 401
        assert client.post("/model/reload", headers={"X-API-Key": "test-key"}).status_code == 200


def test_price_endpoint_recommends_no_change_at_the_default_elasticity(no_registry, monkeypatch):
    monkeypatch.setenv("DFP_API_KEYS", "test-key")
    reset_settings_cache()
    with _client() as client:
        r = client.post(
            "/price/recommend",
            json={"hotel": "City Hotel", "base_adr": 120, "expected_demand": 37.3},
            headers={"X-API-Key": "test-key"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["elasticity"] == -1.0 and body["elasticity_source"].startswith("default")
    assert body["recommended_change"] == 0.0 and body["recommended_price"] == 120
    assert "no change" in body["note"]
