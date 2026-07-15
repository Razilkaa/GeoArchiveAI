from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rag_service import ReportAnswerService, load_report_configs


PROJECT_ROOT = Path(r"C:\FINAM\Conference")
RUNS_ROOT = PROJECT_ROOT / "runs"
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
OCR_API_URL = os.getenv("OCR_API_URL", "http://127.0.0.1:18080")
CONFIGS = load_report_configs()

app = FastAPI(title="GeoArchive API", version="1.0.0")
OCR_SESSION = requests.Session()
OCR_SESSION.trust_env = False


class AskRequest(BaseModel):
    question: str
    mode: Literal["live", "saved", "retrieval_only"] = "live"


class RunRequest(BaseModel):
    force: list[str] = Field(default_factory=list)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def run_dir(report_id: str) -> Path:
    if report_id in {".", ".."} or Path(report_id).name != report_id:
        raise HTTPException(400, "invalid_report_id")
    path = (RUNS_ROOT / report_id).resolve()
    if path.parent != RUNS_ROOT.resolve():
        raise HTTPException(400, "invalid_report_id")
    if not (path / "job.json").exists():
        raise HTTPException(404, "report_not_found")
    return path


@lru_cache(maxsize=4)
def answer_service(report_id: str) -> ReportAnswerService:
    if report_id not in CONFIGS:
        raise KeyError(report_id)
    return ReportAnswerService(CONFIGS[report_id])


def report_status_payload(report_id: str) -> dict[str, Any]:
    directory = run_dir(report_id)
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "reports": len(list(RUNS_ROOT.glob("*/job.json")))}


@app.get("/api/services")
def services() -> dict[str, Any]:
    try:
        response = OCR_SESSION.get(f"{OCR_API_URL.rstrip('/')}/health", timeout=4)
        response.raise_for_status()
        ocr = response.json()
    except Exception as error:
        ocr = {"status": "unavailable", "detail": type(error).__name__}
    return {"api": {"status": "ok"}, "ocr": ocr, "ocr_url": OCR_API_URL}


@app.get("/api/reports")
def reports() -> dict[str, Any]:
    items = []
    for manifest_path in sorted(RUNS_ROOT.glob("*/job.json")):
        report_id = manifest_path.parent.name
        payload = report_status_payload(report_id)
        items.append(
            {
                "report_id": report_id,
                "source_root": payload["source_root"],
                "summary": payload["summary"],
                "worker_status": payload["worker"].get("status", "pending"),
                "operator_status": payload["operator"].get("status", "pending"),
            }
        )
    return {"reports": items, "count": len(items)}


@app.get("/api/reports/{report_id}/status")
def report_status(report_id: str) -> dict[str, Any]:
    return report_status_payload(report_id)


@app.get("/api/reports/{report_id}/artifacts")
def report_artifacts(report_id: str) -> dict[str, Any]:
    directory = run_dir(report_id)
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
    return {"report_id": report_id, "artifacts": artifacts}


@app.post("/api/reports/{report_id}/run", status_code=202)
def run_report(report_id: str, request: RunRequest) -> dict[str, Any]:
    directory = run_dir(report_id)
    lock = directory / "worker" / "worker.lock"
    if lock.exists():
        raise HTTPException(409, "report_already_running")
    allowed = {"page_routing", "fast_ocr", "markdown", "ragflow", "agents", "bundle", "operator"}
    invalid = set(request.force) - allowed
    if invalid:
        raise HTTPException(400, f"invalid_force_stages:{','.join(sorted(invalid))}")
    command = [
        sys.executable,
        str(PIPELINE_ROOT / "job_worker.py"),
        report_id,
        "--ocr-api-url",
        OCR_API_URL,
    ]
    for stage in request.force:
        command.extend(["--force", stage])
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    stdout = (directory / "worker.api.stdout.log").open("ab")
    stderr = (directory / "worker.api.stderr.log").open("ab")
    process = subprocess.Popen(
        command,
        cwd=PIPELINE_ROOT,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    stdout.close()
    stderr.close()
    return {"report_id": report_id, "status": "accepted", "pid": process.pid, "force": request.force}


@app.get("/api/reports/{report_id}")
def report_bundle(report_id: str) -> dict[str, Any]:
    try:
        return answer_service(report_id).bundle
    except KeyError:
        directory = run_dir(report_id)
        result = read_json(directory / "orchestrator" / "report_result.json")
        if result is None:
            raise HTTPException(409, "report_not_ready")
        return result


@app.post("/api/reports/{report_id}/ask")
async def ask(report_id: str, request: AskRequest) -> dict[str, Any]:
    try:
        service = answer_service(report_id)
    except KeyError as error:
        run_dir(report_id)
        raise HTTPException(409, "report_search_not_ready") from error
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "empty_question")
    if request.mode == "saved":
        result = service.saved_answer(question)
        if result is None:
            raise HTTPException(404, "saved_answer_not_found")
        return result
    try:
        return await run_in_threadpool(service.ask, question, request.mode == "live")
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(502, f"upstream_failure:{type(error).__name__}") from error
