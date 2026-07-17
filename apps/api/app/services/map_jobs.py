from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException

from app.config import settings


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PIPELINE_ARTIFACTS = {
    "ocr",
    "trace_preview",
    "assignment_preview",
    "surface_preview",
    "pixel_grid",
    "pixel_contours",
}
GEOREFERENCE_ARTIFACTS = {"cps3", "xyz", "prj", "georeference_metadata"}


def map_jobs_root() -> Path:
    root = settings.runs_root / "map_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def map_job_dir(job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(404, "map_job_not_found")
    directory = map_jobs_root() / job_id
    if not directory.exists():
        raise HTTPException(404, "map_job_not_found")
    return directory


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def start_map_job(
    filename: str,
    source: BinaryIO,
    *,
    interval: float | None = None,
    trace_scale: float = 0.6,
) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "unsupported_map_image")
    if trace_scale <= 0 or trace_scale > 1:
        raise HTTPException(400, "invalid_trace_scale")
    job_id = uuid.uuid4().hex
    directory = map_jobs_root() / job_id
    directory.mkdir(parents=True)
    input_path = directory / f"input{suffix}"
    size = 0
    with input_path.open("wb") as destination:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                destination.close()
                input_path.unlink(missing_ok=True)
                directory.rmdir()
                raise HTTPException(413, "map_image_too_large")
            destination.write(chunk)

    command = [
        sys.executable,
        "-m",
        "services.map_digitizer.pipeline",
        str(input_path),
        "--output-dir",
        str(directory),
        "--ocr-api-url",
        settings.ocr_api_url,
        "--trace-scale",
        str(trace_scale),
    ]
    if interval is not None:
        command.extend(["--interval", str(interval)])
    stdout = (directory / "worker.stdout.log").open("ab")
    stderr = (directory / "worker.stderr.log").open("ab")
    process = subprocess.Popen(
        command,
        cwd=settings.project_root,
        stdout=stdout,
        stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout.close()
    stderr.close()
    job = {
        "job_id": job_id,
        "status": "running",
        "pid": process.pid,
        "filename": Path(filename).name,
        "size_bytes": size,
        "interval": interval,
        "trace_scale": trace_scale,
    }
    (directory / "job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return job


def map_job_status(job_id: str) -> dict:
    directory = map_job_dir(job_id)
    job = read_json(directory / "job.json") or {"job_id": job_id}
    result = read_json(directory / "pipeline_result.json")
    if result is not None:
        urls = {
            name: f"/api/maps/jobs/{job_id}/artifacts/{name}"
            for name in PIPELINE_ARTIFACTS
            if result.get("artifacts", {}).get(name)
        }
        if list((directory / "georeferenced").glob("*.json")):
            urls.update(
                {
                    name: f"/api/maps/jobs/{job_id}/artifacts/{name}"
                    for name in GEOREFERENCE_ARTIFACTS
                }
            )
        return {
            **job,
            "status": result.get("status"),
            "result": result,
            "artifact_urls": urls,
        }
    pid = job.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return {**job, "status": "failed", "error": "worker_exited_without_manifest"}
    return job


def map_job_artifact(job_id: str, artifact_name: str) -> Path:
    directory = map_job_dir(job_id).resolve()
    candidate: Path | None = None
    if artifact_name in PIPELINE_ARTIFACTS:
        result = read_json(directory / "pipeline_result.json") or {}
        raw_path = result.get("artifacts", {}).get(artifact_name)
        if raw_path:
            candidate = Path(raw_path)
    elif artifact_name in GEOREFERENCE_ARTIFACTS:
        manifests = sorted(
            (directory / "georeferenced").glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if manifests:
            metadata_path = manifests[0]
            if artifact_name == "georeference_metadata":
                candidate = metadata_path
            else:
                metadata = read_json(metadata_path) or {}
                raw_path = metadata.get("files", {}).get(artifact_name)
                if raw_path:
                    candidate = Path(raw_path)
    else:
        raise HTTPException(404, "map_artifact_not_found")

    if candidate is None:
        raise HTTPException(409, "map_artifact_not_ready")
    if not candidate.is_absolute():
        candidate = directory / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(directory) or not candidate.is_file():
        raise HTTPException(404, "map_artifact_not_found")
    return candidate


def georeference_map_job(
    job_id: str,
    *,
    target_crs: str,
    control_points: list[dict],
    cell_size: float | None,
    name: str,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
        raise HTTPException(400, "invalid_surface_name")
    directory = map_job_dir(job_id)
    status = map_job_status(job_id)
    result = status.get("result")
    if not result or result.get("status") not in {"accepted", "review"}:
        raise HTTPException(409, "map_surface_not_ready")
    grid_path = Path(result.get("artifacts", {}).get("pixel_grid", ""))
    if not grid_path.is_absolute():
        grid_path = settings.project_root / grid_path
    if not grid_path.exists():
        raise HTTPException(409, "pixel_grid_not_found")
    controls_path = directory / "control_points.json"
    controls_path.write_text(
        json.dumps(control_points, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_dir = directory / "georeferenced"
    command = [
        sys.executable,
        "-m",
        "services.map_digitizer.georeference_grid",
        str(grid_path),
        str(controls_path),
        "--output-dir",
        str(output_dir),
        "--target-crs",
        target_crs,
        "--name",
        name,
    ]
    if cell_size is not None:
        command.extend(["--cell-size", str(cell_size)])
    completed = subprocess.run(
        command,
        cwd=settings.project_root,
        capture_output=True,
        timeout=180,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    (directory / "georeference.stdout.log").write_bytes(completed.stdout)
    (directory / "georeference.stderr.log").write_bytes(completed.stderr)
    if completed.returncode:
        raise HTTPException(422, "georeference_failed")
    metadata = read_json(output_dir / f"{name}.json")
    if metadata is None:
        raise HTTPException(500, "georeference_manifest_missing")
    return {
        **metadata,
        "artifact_urls": {
            artifact: f"/api/maps/jobs/{job_id}/artifacts/{artifact}"
            for artifact in GEOREFERENCE_ARTIFACTS
        },
    }
