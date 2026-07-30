from fastapi.testclient import TestClient

from app.main import app


def test_openapi_exposes_artifact_and_rag_contracts() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/reports/{report_id}/artifacts/{artifact_id}" in paths
    assert "/api/reports/{report_id}/maps/artifacts/{artifact_id}" in paths
    assert "/api/reports/{report_id}/maps/artifacts/{artifact_id}/preview" in paths
    assert "/api/reports/{report_id}/ask" in paths
    assert "/api/search" in paths
