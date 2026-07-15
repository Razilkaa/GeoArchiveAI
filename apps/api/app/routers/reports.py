from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import AskRequest, IntakeResponse, RunRequest
from app.services.intake import register_inbox, save_upload
from app.services.jobs import start_report
from app.services.reports import artifact_payload, list_reports, read_json, report_run_dir, status_payload
from rag_service import ReportAnswerService, load_report_configs


router = APIRouter(prefix="/api", tags=["reports"])
configs = load_report_configs()


@lru_cache(maxsize=4)
def answer_service(report_id: str) -> ReportAnswerService:
    if report_id not in configs:
        raise KeyError(report_id)
    return ReportAnswerService(configs[report_id])


@router.get("/reports")
def reports() -> dict[str, Any]:
    items = list_reports()
    return {"reports": items, "count": len(items)}


@router.post("/intake/scan", response_model=IntakeResponse)
async def scan_inbox() -> IntakeResponse:
    summary = await run_in_threadpool(register_inbox)
    registered = summary.get("registered", [])
    return IntakeResponse(
        status="completed",
        registered=len(registered),
        report_ids=[str(item["report_id"]) for item in registered],
        unsupported=[str(item) for item in summary.get("unsupported", [])],
    )


@router.post("/reports/upload", response_model=IntakeResponse)
async def upload_report(file: UploadFile = File(...)) -> IntakeResponse:
    saved = await run_in_threadpool(save_upload, file.filename or "archive.zip", file.file)
    summary = await run_in_threadpool(register_inbox)
    registered = summary.get("registered", [])
    return IntakeResponse(
        status="completed",
        registered=len(registered),
        report_ids=[str(item["report_id"]) for item in registered],
        unsupported=[str(item) for item in summary.get("unsupported", [])],
        uploaded_name=saved.name,
    )


@router.get("/reports/{report_id}/status")
def report_status(report_id: str) -> dict[str, Any]:
    return status_payload(report_id)


@router.get("/reports/{report_id}/artifacts")
def report_artifacts(report_id: str) -> dict[str, Any]:
    return {"report_id": report_id, "artifacts": artifact_payload(report_id)}


@router.post("/reports/{report_id}/run", status_code=202)
def run_report(report_id: str, request: RunRequest) -> dict[str, Any]:
    return start_report(report_id, request.force)


@router.get("/reports/{report_id}")
def report_bundle(report_id: str) -> dict[str, Any]:
    directory = report_run_dir(report_id)
    result = read_json(directory / "result_bundle.json") or read_json(
        directory / "orchestrator" / "report_result.json"
    )
    if result is not None:
        return result
    try:
        return answer_service(report_id).bundle
    except KeyError:
        raise HTTPException(409, "report_not_ready")


@router.post("/reports/{report_id}/ask")
async def ask(report_id: str, request: AskRequest) -> dict[str, Any]:
    try:
        service = answer_service(report_id)
    except KeyError as error:
        report_run_dir(report_id)
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
