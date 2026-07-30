from dataclasses import replace

import requests

import rag_service


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"code": 0, "data": {"chunks": [], "total": 0}}


class FlakySession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise requests.Timeout("temporary")
        return FakeResponse()


def test_ragflow_retries_transient_timeout(monkeypatch) -> None:
    session = FlakySession()
    monkeypatch.setattr(
        rag_service,
        "settings",
        replace(
            rag_service.settings,
            ragflow_retrieval_attempts=2,
            ragflow_retry_backoff_s=0,
        ),
    )
    client = rag_service.RagflowClient(
        token="token",
        dataset_id="dataset",
        base_url="https://ragflow.example/api/v1",
        session=session,
    )

    result = client.retrieve("question")

    assert session.calls == 2
    assert result["chunks"] == []


class CapturingSession:
    def __init__(self) -> None:
        self.payload = None

    def post(self, *args, **kwargs):
        self.payload = kwargs["json"]
        return FakeResponse()


def test_ragflow_does_not_request_slow_keyword_generation() -> None:
    session = CapturingSession()
    client = rag_service.RagflowClient(
        token="token",
        dataset_id="dataset",
        base_url="https://ragflow.example/api/v1",
        session=session,
    )

    client.retrieve("question")

    assert session.payload["keyword"] is False
