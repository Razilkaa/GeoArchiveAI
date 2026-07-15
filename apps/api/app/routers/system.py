from __future__ import annotations

from typing import Any

import requests
from fastapi import APIRouter

from app.config import settings
from app.services.automation import queue_status


router = APIRouter(tags=["system"])
ocr_session = requests.Session()
ocr_session.trust_env = False


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "reports": len(list(settings.runs_root.glob("*/job.json")))}


@router.get("/api/services")
def services() -> dict[str, Any]:
    try:
        health_response = ocr_session.get(f"{settings.ocr_api_url.rstrip('/')}/health", timeout=4)
        health_response.raise_for_status()
        metrics_response = ocr_session.get(f"{settings.ocr_api_url.rstrip('/')}/metrics", timeout=4)
        metrics_response.raise_for_status()
        ocr = {**health_response.json(), "metrics": metrics_response.json()}
    except requests.RequestException as error:
        ocr = {"status": "unavailable", "detail": type(error).__name__}
    return {
        "api": {"status": "ok", "docs": "/docs"},
        "ocr": ocr,
        "ragflow": {"url": settings.ragflow_url},
    }


@router.get("/api/queue")
def queue() -> dict[str, Any]:
    return queue_status()
