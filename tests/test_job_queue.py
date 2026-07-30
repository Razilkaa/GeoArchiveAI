import json

from app.services import jobs
from geoarchive.settings import Settings


def test_api_queues_report_atomically(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEOARCHIVE_ROOT", str(tmp_path / "app"))
    monkeypatch.setenv("GEOARCHIVE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("JOB_EXECUTION_MODE", "queue")
    monkeypatch.setenv("RAGFLOW_URL", "https://ragflow.example/api/v1")
    settings = Settings.from_environment()
    report_dir = settings.runs_root / "384092"
    report_dir.mkdir(parents=True)
    (report_dir / "job.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(jobs, "settings", settings)

    result = jobs.start_report("384092", ["bundle"])

    assert result["status"] == "queued"
    queued = list(settings.queue_root.glob("*.json"))
    assert len(queued) == 1
    payload = json.loads(queued[0].read_text(encoding="utf-8"))
    assert payload["report_id"] == "384092"
    assert payload["force"] == ["bundle"]
    assert not list(settings.queue_root.glob("*.part"))
