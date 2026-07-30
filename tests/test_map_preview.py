import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from app.services import reports as reports_service


def test_map_preview_is_browser_sized_and_cached(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_root = tmp_path / "runs"
    source_root = tmp_path / "source"
    report_dir = runs_root / "demo"
    source_root.mkdir()
    report_dir.mkdir(parents=True)
    source = source_root / "map.png"
    Image.new("RGB", (3000, 2000), "white").save(source)
    (report_dir / "job.json").write_text(
        json.dumps(
            {
                "report_id": "demo",
                "source_root": str(source_root),
                "pages": [
                    {
                        "id": "page:00001",
                        "relative_path": source.name,
                        "content_type": "map",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reports_service,
        "settings",
        replace(
            reports_service.settings,
            runs_root=runs_root,
            project_root=tmp_path,
        ),
    )

    payload = reports_service.map_payload("demo")
    artifact = payload["sources"][0]
    assert artifact["preview_url"].endswith("/source-0/preview")

    first = reports_service.report_map_artifact_preview("demo", "source-0")
    second = reports_service.report_map_artifact_preview("demo", "source-0")
    assert first == second
    with Image.open(first) as preview:
        assert preview.size == (1600, 1067)
