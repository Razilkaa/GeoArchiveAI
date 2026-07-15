from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.config import settings


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
        lock_exists = (manifest_path.parent / "worker" / "worker.lock").exists()
        items.append(
            {
                "report_id": report_id,
                "source_root": payload["source_root"],
                "summary": payload["summary"],
                "worker_status": "running" if lock_exists else payload["worker"].get("status", "pending"),
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
                "media_type": f"image/{path.suffix.casefold().lstrip('.') or 'jpeg'}",
                "exists": path.exists(),
            }
        )

    result = read_json(directory / "map_agent" / "result.json", {})
    digitized = []
    for artifact in result.get("artifacts", []):
        if artifact.get("name") == "source_map":
            continue
        path = Path(str(artifact.get("path") or ""))
        digitized.append(
            {
                **artifact,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
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
