from fastapi.testclient import TestClient
from app.main import app


def test_demo_run_is_immutable_and_has_null_retrieval_metric():
    with TestClient(app) as client:
        first = client.post("/api/demo/run?target_id=mock-a").json()
        second = client.post("/api/demo/run?target_id=mock-a").json()
        assert first["id"] != second["id"]
        details = client.get(f"/api/runs/{first['id']}").json()
        assert details["status"] == "completed"
        assert details["metrics"]["retrieval_recall_at_k"]["value"] is None


def test_local_endpoint_is_rejected():
    with TestClient(app) as client:
        response = client.post("/api/targets", json={"name": "Unsafe", "adapter_type": "generic_http_json", "endpoint": "http://127.0.0.1:8000", "version": "1", "languages": ["ru"]})
        assert response.status_code == 422
