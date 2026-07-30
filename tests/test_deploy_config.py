from pathlib import Path

import yaml


def test_compose_defines_isolated_web_api_worker_and_ocr() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((root / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    services = payload["services"]
    assert set(services) == {"web", "api", "worker", "ocr"}
    assert services["api"]["environment"]["JOB_EXECUTION_MODE"] == "queue"
    assert services["worker"]["command"] == ["python", "-m", "pipeline.queue_worker"]
    assert services["web"]["depends_on"]["api"]["condition"] == "service_healthy"
