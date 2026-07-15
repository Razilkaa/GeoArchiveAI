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
