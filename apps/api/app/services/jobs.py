from __future__ import annotations

import subprocess
import sys
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
    "agents",
    "bundle",
    "operator",
}


def start_report(report_id: str, force: list[str]) -> dict[str, Any]:
    directory = report_run_dir(report_id)
    lock = directory / "worker" / "worker.lock"
    if lock.exists():
        raise HTTPException(409, "report_already_running")
    invalid = set(force) - ALLOWED_FORCE_STAGES
    if invalid:
        raise HTTPException(400, f"invalid_force_stages:{','.join(sorted(invalid))}")

    command = [
        sys.executable,
        str(settings.pipeline_root / "job_worker.py"),
        report_id,
        "--ocr-api-url",
        settings.ocr_api_url,
        "--proxy",
        settings.proxy_url,
    ]
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
