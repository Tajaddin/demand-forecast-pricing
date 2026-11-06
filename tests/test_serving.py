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
