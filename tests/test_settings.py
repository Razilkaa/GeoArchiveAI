from pathlib import Path

from geoarchive.settings import Settings


def test_settings_derive_runtime_paths_from_data_root(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "app"
    data = tmp_path / "data"
    monkeypatch.setenv("GEOARCHIVE_ROOT", str(project))
    monkeypatch.setenv("GEOARCHIVE_DATA_ROOT", str(data))
    monkeypatch.setenv("RAGFLOW_URL", "https://ragflow.example/api/v1/")
    monkeypatch.setenv("RAGFLOW_PROXY", "off")
    monkeypatch.setenv("RAGFLOW_DATASET_ID", "dataset-test")
    monkeypatch.setenv("ENABLE_ENRICHMENT", "true")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1234")

    settings = Settings.from_environment()

    assert settings.project_root == project.resolve()
    assert settings.runs_root == (data / "runs").resolve()
    assert settings.results_root == (data / "results").resolve()
    assert settings.ragflow_url == "https://ragflow.example/api/v1"
    assert settings.ragflow_dataset_id == "dataset-test"
    assert settings.enable_enrichment is True
    assert settings.proxy_url is None
    assert settings.max_upload_bytes == 1234


def test_invalid_job_execution_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("JOB_EXECUTION_MODE", "unknown")
    try:
        Settings.from_environment()
    except ValueError as error:
        assert "JOB_EXECUTION_MODE" in str(error)
    else:
        raise AssertionError("invalid mode must fail")
