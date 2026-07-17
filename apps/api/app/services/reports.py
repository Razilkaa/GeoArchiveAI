from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.config import settings


BASE_STAGES = ("page_routing", "fast_ocr", "markdown", "ragflow", "bundle")
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".pdf": "application/pdf",
    ".geojson": "application/geo+json",
    ".json": "application/json",
    ".cps3": "text/plain",
    ".xyz": "text/plain",
    ".prj": "text/plain",
    ".npz": "application/octet-stream",
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def report_run_dir(report_id: str) -> Path:
    if report_id in {".", ".."} or Path(report_id).name != report_id:
        raise HTTPException(400, "invalid_report_id")
    path = (settings.runs_root / report_id).resolve()
    if path.parent != settings.runs_root.resolve():
        raise HTTPException(400, "invalid_report_id")
    if not (path / "job.json").exists():
        raise HTTPException(404, "report_not_found")
    return path


def status_payload(report_id: str) -> dict[str, Any]:
    directory = report_run_dir(report_id)
    manifest = read_json(directory / "job.json", {})
    worker = read_json(directory / "worker" / "state.json", {})
    lock_exists = (directory / "worker" / "worker.lock").exists()
    if lock_exists:
        worker["status"] = "running"
    elif worker.get("status") == "running" and all(
        worker.get("stages", {}).get(name, {}).get("status") == "completed"
        for name in BASE_STAGES
    ):
        worker["status"] = "completed"
    operator = read_json(directory / "orchestrator" / "state.json", {})
    return {
        "report_id": report_id,
        "source_root": manifest.get("source_root"),
        "summary": manifest.get("summary", {}),
        "worker": worker,
        "operator": operator,
    }


def list_reports() -> list[dict[str, Any]]:
    items = []
    for manifest_path in sorted(settings.runs_root.glob("*/job.json")):
        report_id = manifest_path.parent.name
        payload = status_payload(report_id)
        items.append(
            {
                "report_id": report_id,
                "source_root": payload["source_root"],
                "summary": payload["summary"],
                "worker_status": payload["worker"].get("status", "pending"),
                "operator_status": payload["operator"].get("status", "pending"),
            }
        )
    return items


def artifact_payload(report_id: str) -> list[dict[str, Any]]:
    directory = report_run_dir(report_id)
    operator = read_json(directory / "orchestrator" / "state.json", {})
    artifacts = []
    for name, record in operator.get("agents", {}).items():
        for artifact in record.get("result", {}).get("artifacts", []):
            path = Path(str(artifact.get("path") or ""))
            artifacts.append(
                {
                    "agent": name,
                    **artifact,
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                }
            )
    contact = directory / "graphics_contact_sheet.jpg"
    if contact.exists():
        artifacts.append(
            {
                "agent": "intake",
                "name": "graphics_contact_sheet",
                "path": str(contact),
                "media_type": "image/jpeg",
                "exists": True,
                "size_bytes": contact.stat().st_size,
            }
        )
    return artifacts


def map_payload(report_id: str) -> dict[str, Any]:
    directory = report_run_dir(report_id)
    manifest = read_json(directory / "job.json", {})
    source_root = Path(str(manifest.get("source_root") or ""))
    sources = []
    for page in manifest.get("pages", []):
        if page.get("content_type") != "map":
            continue
        path = source_root / str(page.get("relative_path") or "")
        sources.append(
            {
                "page_id": page.get("id"),
                "label": f"Исходная карта · {Path(str(page.get('relative_path') or '')).name}",
                "path": str(path),
                "media_type": MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream"),
                "exists": path.exists(),
            }
        )

    result = read_json(directory / "map_agent" / "result.json", {})
    digitized = []
    for artifact in result.get("artifacts", []):
        if artifact.get("name") == "source_map":
            path = Path(str(artifact.get("path") or ""))
            if path.exists() and not any(item["path"] == str(path) for item in sources):
                sources.insert(
                    0,
                    {
                        "page_id": None,
                        "label": artifact.get("label") or f"Исходная карта · {path.name}",
                        "path": str(path),
                        "media_type": artifact.get("media_type")
                        or MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream"),
                        "exists": True,
                    }
                )
            continue
        path = Path(str(artifact.get("path") or ""))
        digitized.append(
            {
                **artifact,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )
    for kind, records in (("source", sources), ("digitized", digitized)):
        for index, item in enumerate(records):
            artifact_id = f"{kind}-{index}"
            item["artifact_id"] = artifact_id
            item["download_url"] = (
                f"/api/reports/{report_id}/maps/artifacts/{artifact_id}"
            )
    return {
        "report_id": report_id,
        "sources": sources,
        "digitized": digitized,
        "status": result.get("status", "not_digitized"),
        "quality_status": result.get("quality_status"),
        "metrics": result.get("metrics", {}),
        "issues": result.get("issues", []),
    }


def report_map_artifact(report_id: str, artifact_id: str) -> tuple[Path, str]:
    directory = report_run_dir(report_id).resolve()
    manifest = read_json(directory / "job.json", {})
    source_root = Path(str(manifest.get("source_root") or "")).resolve()
    payload = map_payload(report_id)
    records = [*payload["sources"], *payload["digitized"]]
    record = next(
        (item for item in records if item.get("artifact_id") == artifact_id), None
    )
    if record is None:
        raise HTTPException(404, "report_map_artifact_not_found")
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute():
        path = settings.project_root / path
    path = path.resolve()
    if not path.is_file() or not (
        path.is_relative_to(directory) or path.is_relative_to(source_root)
    ):
        raise HTTPException(404, "report_map_artifact_not_found")
    media_type = str(
        record.get("media_type")
        or MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream")
    )
    return path, media_type
