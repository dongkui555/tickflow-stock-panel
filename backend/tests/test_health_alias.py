from fastapi.testclient import TestClient

from app.main import app


def test_api_health_returns_the_same_json_as_health(monkeypatch):
    monkeypatch.setattr("app.api.routes._memory_snapshot", lambda: {"rss_bytes": 4096})
    client = TestClient(app)
    health = client.get("/health")
    alias = client.get("/api/health")

    assert health.status_code == alias.status_code == 200
    assert alias.headers["content-type"].startswith("application/json")
    assert alias.json() == health.json()
    assert alias.json()["status"] == "ok"
