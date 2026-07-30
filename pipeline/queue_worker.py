from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geoarchive.settings import Settings, load_settings
from geoarchive.object_storage import ObjectStorage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_command(request: dict[str, Any], settings: Settings) -> list[str]:
    command = [
        sys.executable,
        str(settings.pipeline_root / "job_worker.py"),
        str(request["report_id"]),
        "--runs-root",
        str(settings.runs_root),
        "--credentials-file",
        str(settings.llm_credentials_path),
        "--token-file",
        str(settings.ragflow_token_path),
        "--ragflow-base-url",
        settings.ragflow_url,
        "--ocr-api-url",
        settings.ocr_api_url,
        "--vision-model",
        settings.llm_model,
        "--agent-model",
        settings.llm_model,
    ]
    if settings.proxy_url:
        command.extend(["--proxy", settings.proxy_url])
    if settings.ragflow_dataset_id:
        command.extend(["--dataset-id", settings.ragflow_dataset_id])
    if settings.enable_enrichment:
        command.append("--enable-enrichment")
    for stage in request.get("force", []):
        command.extend(["--force", str(stage)])
    return command


def process_one(settings: Settings) -> bool:
    queue_root = settings.queue_root
    processing_root = queue_root / "processing"
    completed_root = queue_root / "completed"
    failed_root = queue_root / "failed"
    for directory in (queue_root, processing_root, completed_root, failed_root):
        directory.mkdir(parents=True, exist_ok=True)

    for queued_path in sorted(queue_root.glob("*.json")):
        claimed_path = processing_root / queued_path.name
        try:
            os.replace(queued_path, claimed_path)
        except (FileNotFoundError, PermissionError):
            continue

        request = json.loads(claimed_path.read_text(encoding="utf-8"))
        command = build_command(request, settings)
        started_at = utc_now()
        completed = subprocess.run(
            command,
            cwd=settings.pipeline_root,
            check=False,
        )
        storage_error = None
        if completed.returncode == 0:
            storage = ObjectStorage(settings)
            if storage.enabled:
                try:
                    report_id = str(request["report_id"])
                    prefix = settings.s3_results_prefix or "results"
                    storage.upload_tree(
                        settings.runs_root / report_id,
                        f"{prefix}/{report_id}/run",
                    )
                    storage.upload_tree(
                        settings.results_root / report_id,
                        f"{prefix}/{report_id}/published",
                    )
                except Exception as error:
                    storage_error = type(error).__name__
                    completed = subprocess.CompletedProcess(
                        completed.args,
                        70,
                        completed.stdout,
                        completed.stderr,
                    )
        result = {
            **request,
            "worker": socket.gethostname(),
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
        }
        if storage_error:
            result["storage_error"] = storage_error
        destination_root = completed_root if completed.returncode == 0 else failed_root
        result_path = destination_root / claimed_path.name
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        claimed_path.unlink(missing_ok=True)
        return True
    return False


def run_forever(settings: Settings) -> None:
    while True:
        if not process_one(settings):
            time.sleep(max(0.2, settings.queue_poll_interval_s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume GeoArchiveAI report jobs")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    if args.once:
        process_one(settings)
    else:
        run_forever(settings)


if __name__ == "__main__":
    main()
