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


def resolved_artifact_path(raw_path: Any) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute():
        path = settings.project_root / path
    return path.resolve()


def report_run_dir(report_id: str) -> Path:
    if report_id in {".", ".."} or Path(report_id).name != report_id:
        raise HTTPException(400, "invalid_report_id")
    path = (settings.runs_root / report_id).resolve()
    if path.parent != settings.runs_root.resolve():
        raise HTTPException(400, "invalid_report_id")
    if not (path / "job.json").exists():
        raise HTTPException(404, "report_not_found")
    return path


def related_map_run_dirs(report_id: str, source_root: Path) -> list[Path]:
    """Return auxiliary map-validation runs belonging to the same report."""
    directories = [report_run_dir(report_id)]
    prefix = f"{report_id}_map_"
    expected_root = source_root.resolve()
    for manifest_path in settings.runs_root.glob(f"{prefix}*/job.json"):
        manifest = read_json(manifest_path, {})
        candidate_root = Path(str(manifest.get("source_root") or ""))
        if candidate_root and candidate_root.resolve() == expected_root:
            directories.append(manifest_path.parent.resolve())
    return directories


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
        if "_map_" in report_id and (settings.runs_root / report_id.split("_map_", 1)[0] / "job.json").exists():
            continue
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
            path = resolved_artifact_path(artifact.get("path"))
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
    for index, artifact in enumerate(artifacts):
        artifact_id = f"artifact-{index}"
        artifact["artifact_id"] = artifact_id
        artifact["download_url"] = f"/api/reports/{report_id}/artifacts/{artifact_id}"
    return artifacts


def report_artifact(report_id: str, artifact_id: str) -> tuple[Path, str]:
    directory = report_run_dir(report_id).resolve()
    records = artifact_payload(report_id)
    record = next((item for item in records if item.get("artifact_id") == artifact_id), None)
    if record is None:
        raise HTTPException(404, "report_artifact_not_found")
    path = resolved_artifact_path(record.get("path"))
    manifest = read_json(directory / "job.json", {})
    source_value = manifest.get("source_root")
    source_root = Path(str(source_value)).resolve() if source_value else None
    allowed = path.is_relative_to(directory) or (
        source_root is not None and path.is_relative_to(source_root)
    )
    if not allowed or not path.is_file():
        raise HTTPException(404, "report_artifact_not_found")
    media_type = str(
        record.get("media_type")
        or MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream")
    )
    return path, media_type


def map_payload(report_id: str) -> dict[str, Any]:
    directory = report_run_dir(report_id)
    bundle = read_json(directory / "result_bundle.json", {})
    manifest = read_json(directory / "job.json", {})
    source_root = Path(str(manifest.get("source_root") or ""))
    related_directories = related_map_run_dirs(report_id, source_root)
    manifests = [read_json(item / "job.json", {}) for item in related_directories]
    results = [read_json(item / "map_agent" / "result.json", {}) for item in related_directories]
    result = results[0] if results else {}
    jobs = [job for payload in results for job in payload.get("jobs", [])]
    processed_status = {
        str(job.get("page_id")): str(job.get("status"))
        for job in jobs
    }
    quality_by_page = {
        str(job.get("page_id")): dict(job.get("quality") or {})
        for job in jobs
    }
    routing_by_page = {
        str(job.get("page_id")): str(job.get("routing_type") or "")
        for job in jobs
    }
    sources = []
    pages = []
    seen_pages = set()
    for payload in manifests:
        for page in payload.get("pages", []):
            key = (str(page.get("id")), str(page.get("relative_path")))
            if key not in seen_pages:
                pages.append(page)
                seen_pages.add(key)
    for page in pages:
        if page.get("content_type") not in {"map", "chart"}:
            continue
        path = source_root / str(page.get("relative_path") or "")
        sources.append(
            {
                "page_id": page.get("id"),
                "label": f"Исходная карта · {Path(str(page.get('relative_path') or '')).name}",
                "path": str(path),
                "media_type": MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream"),
                "exists": path.exists(),
                "status": processed_status.get(str(page.get("id")), "pending"),
                "quality": quality_by_page.get(str(page.get("id")), {}),
                "content_type": page.get("content_type"),
            }
        )

    digitized = []
    seen_artifacts = set()
    for artifact in [item for payload in results for item in payload.get("artifacts", [])]:
        artifact_path = str(resolved_artifact_path(artifact.get("path")))
        if artifact_path in seen_artifacts:
            continue
        seen_artifacts.add(artifact_path)
        if artifact.get("name") == "source_map":
            path = resolved_artifact_path(artifact.get("path"))
            if path.exists() and not any(item["path"] == str(path) for item in sources):
                sources.insert(
                    0,
                    {
                        "page_id": artifact.get("page_id"),
                        "label": artifact.get("label") or f"Исходная карта · {path.name}",
                        "path": str(path),
                        "media_type": artifact.get("media_type")
                        or MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream"),
                        "exists": True,
                        "status": processed_status.get(str(artifact.get("page_id")), "pending"),
                        "quality": quality_by_page.get(str(artifact.get("page_id")), {}),
                        "content_type": routing_by_page.get(str(artifact.get("page_id"))),
                    }
                )
            continue
        path = resolved_artifact_path(artifact.get("path"))
        digitized.append(
            {
                **artifact,
                "quality": quality_by_page.get(str(artifact.get("page_id")), {}),
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
        "quality_status": "review" if any(item.get("quality_status") == "review" for item in results) else result.get("quality_status"),
        "metrics": {
            **result.get("metrics", {}),
            "related_runs": len(related_directories),
            "digitized_pages": len(
                {item.get("page_id") for item in digitized if item.get("page_id")}
            ),
            "source_pages": len(
                {item.get("page_id") for item in sources if item.get("page_id")}
            ),
        },
        "issues": [issue for payload in results for issue in payload.get("issues", [])],
        "structures": (bundle.get("entities") or {}).get("structures", []),
        "horizons": (bundle.get("entities") or {}).get("horizons", []),
    }


def report_map_artifact(report_id: str, artifact_id: str) -> tuple[Path, str]:
    directory = report_run_dir(report_id).resolve()
    manifest = read_json(directory / "job.json", {})
    source_root_value = manifest.get("source_root")
    source_root = Path(str(source_root_value)).resolve() if source_root_value else None
    related_directories = related_map_run_dirs(report_id, source_root) if source_root else [directory]
    payload = map_payload(report_id)
    records = [*payload["sources"], *payload["digitized"]]
    record = next(
        (item for item in records if item.get("artifact_id") == artifact_id), None
    )
    if record is None:
        raise HTTPException(404, "report_map_artifact_not_found")
    path = resolved_artifact_path(record.get("path"))
    if not path.is_file() or not (
        any(path.is_relative_to(item) for item in related_directories)
        or (source_root is not None and path.is_relative_to(source_root))
    ):
        raise HTTPException(404, "report_map_artifact_not_found")
    media_type = str(
        record.get("media_type")
        or MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream")
    )
    return path, media_type
