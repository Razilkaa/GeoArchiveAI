from pathlib import Path

from geoarchive.settings import Settings
from pipeline.queue_worker import build_command


def test_queue_worker_command_uses_shared_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOARCHIVE_ROOT", str(tmp_path / "app"))
    monkeypatch.setenv("GEOARCHIVE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("GEOARCHIVE_SECRETS_ROOT", str(tmp_path / "secrets"))
    monkeypatch.setenv("OCR_API_URL", "http://ocr:8080")
    monkeypatch.setenv("RAGFLOW_URL", "https://ragflow.example/api/v1")
    monkeypatch.setenv("RAGFLOW_PROXY", "off")
    monkeypatch.setenv("ENABLE_ENRICHMENT", "true")
    settings = Settings.from_environment()

    command = build_command(
        {"report_id": "384092", "force": ["ragflow", "bundle"]}, settings
    )

    joined = " ".join(command)
    assert "384092" in command
    assert "http://ocr:8080" in command
    assert "https://ragflow.example/api/v1" in command
    assert "--proxy" not in command
    assert "--enable-enrichment" in command
    assert "--force ragflow" in joined
    assert "--force bundle" in joined
