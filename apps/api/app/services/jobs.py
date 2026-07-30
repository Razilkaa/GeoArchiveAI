from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.config import settings
from app.services.reports import report_run_dir


ALLOWED_FORCE_STAGES = {
    "page_routing",
    "fast_ocr",
    "markdown",
    "ragflow",
    "map_digitization",
    "agents",
    "bundle",
    "operator",
}


def start_report(report_id: str, force: list[str]) -> dict[str, Any]:
    directory = report_run_dir(report_id)
    lock = directory / "worker" / "worker.lock"
    if lock.exists():
        return {"report_id": report_id, "status": "running", "pid": None, "force": force}
    invalid = set(force) - ALLOWED_FORCE_STAGES
    if invalid:
        raise HTTPException(400, f"invalid_force_stages:{','.join(sorted(invalid))}")

    if settings.job_execution_mode == "queue":
        settings.queue_root.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        queued = {
            "schema_version": 1,
            "request_id": request_id,
            "report_id": report_id,
            "force": force,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = settings.queue_root / f"{request_id}.json.part"
        target = settings.queue_root / f"{request_id}.json"
        temporary.write_text(json.dumps(queued, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return {
            "report_id": report_id,
            "status": "queued",
            "request_id": request_id,
            "force": force,
        }

    command = [
        sys.executable,
        str(settings.pipeline_root / "job_worker.py"),
        report_id,
        "--ocr-api-url",
        settings.ocr_api_url,
        "--ragflow-base-url",
        settings.ragflow_url,
        "--runs-root",
        str(settings.runs_root),
        "--credentials-file",
        str(settings.llm_credentials_path),
        "--token-file",
        str(settings.ragflow_token_path),
        "--vision-model",
        settings.llm_model,
        "--agent-model",
        settings.llm_model,
    ]
    if settings.proxy_url:
        command.extend(["--proxy", settings.proxy_url])
    if settings.enable_enrichment:
        command.append("--enable-enrichment")
    for stage in force:
        command.extend(["--force", stage])
    stdout = (directory / "worker.api.stdout.log").open("ab")
    stderr = (directory / "worker.api.stderr.log").open("ab")
    process = subprocess.Popen(
        command,
        cwd=settings.pipeline_root,
        stdout=stdout,
        stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout.close()
    stderr.close()
    return {"report_id": report_id, "status": "accepted", "pid": process.pid, "force": force}
