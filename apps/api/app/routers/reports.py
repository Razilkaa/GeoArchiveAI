from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import AskRequest, IntakeResponse, RunRequest
from app.config import settings
from app.services.automation import reconcile_now
from app.services.intake import save_upload
from app.services.jobs import start_report
from app.services.reports import artifact_payload, list_reports, map_payload, read_json, report_run_dir, status_payload
from rag_service import CorpusAnswerService, ReportAnswerService, ReportConfig, load_report_configs


router = APIRouter(prefix="/api", tags=["reports"])
configs = load_report_configs()


def report_config(report_id: str) -> ReportConfig:
    directory = report_run_dir(report_id)
    dynamic = ReportConfig(
        report_id=report_id,
        bundle_path=directory / "result_bundle.json",
        ragflow_metadata_path=directory / "ragflow.json",
        local_corpus_path=directory / f"report_{report_id}_fast_ocr.md",
        ragflow_token_path=settings.project_root / "secrets" / "ragflow_token.txt",
        llm_credentials_path=settings.project_root / "secrets" / "tokent.txt",
    )
    if all(
        path.exists()
        for path in (
            dynamic.bundle_path,
            dynamic.ragflow_metadata_path,
            dynamic.local_corpus_path,
            dynamic.ragflow_token_path,
            dynamic.llm_credentials_path,
        )
    ):
        return dynamic
    if report_id in configs:
        return configs[report_id]
    raise KeyError(report_id)


def answer_service(report_id: str) -> ReportAnswerService:
    return ReportAnswerService(report_config(report_id))


def corpus_answer_service() -> CorpusAnswerService:
    corpus_configs = []
    for item in list_reports():
        try:
            corpus_configs.append(report_config(str(item["report_id"])))
        except (KeyError, OSError, ValueError):
            continue
    if not corpus_configs:
        raise KeyError("corpus_search_not_ready")
    return CorpusAnswerService(corpus_configs)


@router.get("/reports")
def reports() -> dict[str, Any]:
    items = list_reports()
    return {"reports": items, "count": len(items)}


@router.post("/intake/scan", response_model=IntakeResponse)
async def scan_inbox() -> IntakeResponse:
    summary = await run_in_threadpool(reconcile_now)
    registered = summary.get("registered", [])
    return IntakeResponse(
        status="completed",
        registered=len(registered),
        report_ids=[str(item["report_id"]) for item in registered],
        unsupported=[str(item) for item in summary.get("unsupported", [])],
        started=len(summary.get("started", [])),
        queued=len(summary.get("queued", [])),
    )


@router.post("/reports/upload", response_model=IntakeResponse)
async def upload_report(file: UploadFile = File(...)) -> IntakeResponse:
    saved = await run_in_threadpool(save_upload, file.filename or "archive.zip", file.file)
    summary = await run_in_threadpool(reconcile_now)
    registered = summary.get("registered", [])
    return IntakeResponse(
        status="completed",
        registered=len(registered),
        report_ids=[str(item["report_id"]) for item in registered],
        unsupported=[str(item) for item in summary.get("unsupported", [])],
        uploaded_name=saved.name,
        started=len(summary.get("started", [])),
        queued=len(summary.get("queued", [])),
    )


@router.get("/reports/{report_id}/status")
def report_status(report_id: str) -> dict[str, Any]:
    return status_payload(report_id)


@router.get("/reports/{report_id}/artifacts")
def report_artifacts(report_id: str) -> dict[str, Any]:
    return {"report_id": report_id, "artifacts": artifact_payload(report_id)}


@router.get("/reports/{report_id}/maps")
def report_maps(report_id: str) -> dict[str, Any]:
    return map_payload(report_id)


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


@router.post("/search")
async def search_corpus(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "empty_question")
    try:
        service = corpus_answer_service()
        return await run_in_threadpool(service.ask, question, request.mode == "live")
    except KeyError as error:
        raise HTTPException(409, "corpus_search_not_ready") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(502, f"upstream_failure:{type(error).__name__}") from error
