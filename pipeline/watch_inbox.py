from __future__ import annotations

import argparse
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from batch_intake import intake
from job_worker import JobWorker, WorkerConfig, discover_dataset_id
from report_orchestrator import ReportOrchestrator


def file_is_ready(path: Path) -> bool:
    if not path.is_file() or path.name.casefold().endswith(".policy.json"):
        return True
    if os.name != "nt":
        try:
            with path.open("rb") as handle:
                handle.read(1)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(str(path), 0x80000000, 0, None, 3, 0x80, None)
    invalid_handle = wintypes.HANDLE(-1).value
    if handle == invalid_handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def inbox_is_ready(inbox: Path) -> bool:
    return all(file_is_ready(path) for path in inbox.rglob("*") if path.is_file())


def fingerprint(inbox: Path) -> tuple[tuple[str, int, int, int], ...]:
    records = []
    for item in inbox.iterdir():
        paths = [item] if item.is_file() else (path for path in item.rglob("*") if path.is_file())
        file_count = 0
        total_size = 0
        latest_mtime = 0
        try:
            for path in paths:
                stat = path.stat()
                file_count += 1
                total_size += stat.st_size
                latest_mtime = max(latest_mtime, stat.st_mtime_ns)
            records.append((item.name, file_count, total_size, latest_mtime))
        except OSError:
            # A file changing while we scan makes this observation unstable.
            records.append((item.name, -1, -1, time.time_ns()))
    return tuple(sorted(records))


def append_log(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def initialize_operators(summary: Path) -> list[dict]:
    payload = json.loads(summary.read_text(encoding="utf-8"))
    initialized = []
    for report in payload.get("registered", []):
        manifest = Path(report["manifest"])
        state = ReportOrchestrator(manifest).run()
        initialized.append({"report_id": report["report_id"], "status": state["status"]})
    return initialized


def run_registered_workers(summary: Path, runs: Path) -> list[dict]:
    payload = json.loads(summary.read_text(encoding="utf-8"))
    project_root = Path(__file__).parents[1]
    dataset_id = discover_dataset_id(runs)
    completed = []
    for report in payload.get("registered", []):
        manifest = Path(report["manifest"])
        config = WorkerConfig(
            credentials_file=project_root / "secrets" / "tokent.txt",
            ragflow_token_file=project_root / "secrets" / "ragflow_token.txt",
            dataset_id=dataset_id,
            ocr_api_url="http://127.0.0.1:18080",
        )
        try:
            state = JobWorker(manifest, config).run()
            completed.append({"report_id": report["report_id"], "status": state["status"]})
        except Exception as error:
            completed.append(
                {
                    "report_id": report["report_id"],
                    "status": "failed",
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                }
            )
    return completed


def watch(
    inbox: Path,
    staging: Path,
    runs: Path,
    interval_s: int,
    settle_s: int,
    auto_process: bool = True,
) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    log = runs / "intake_watcher.jsonl"
    observed = None
    observed_since = time.monotonic()
    processed = None
    while True:
        current = fingerprint(inbox)
        now = time.monotonic()
        if current != observed:
            observed = current
            observed_since = now
        elif current != processed and now - observed_since >= settle_s and inbox_is_ready(inbox):
            try:
                summary = intake(inbox, staging, runs)
                payload = json.loads(summary.read_text(encoding="utf-8"))
                operators = initialize_operators(summary)
                workers = run_registered_workers(summary, runs) if auto_process else []
                append_log(
                    log,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "status": "completed",
                        "registered": len(payload["registered"]),
                        "unsupported": payload["unsupported"],
                        "operators": operators,
                        "workers": workers,
                    },
                )
                processed = current
            except Exception as error:
                append_log(
                    log,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "status": "failed",
                        "error": type(error).__name__,
                        "detail": str(error)[:500],
                    },
                )
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch report inbox and run local-only intake")
    parser.add_argument("--inbox", type=Path, default=Path(r"C:\FINAM\Conference\reports_inbox"))
    parser.add_argument("--staging", type=Path, default=Path(r"C:\FINAM\Conference\reports_staging"))
    parser.add_argument("--runs", type=Path, default=Path(r"C:\FINAM\Conference\runs"))
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--settle", type=int, default=60)
    parser.add_argument("--no-auto-process", action="store_true")
    args = parser.parse_args()
    watch(
        args.inbox,
        args.staging,
        args.runs,
        max(5, args.interval),
        max(15, args.settle),
        auto_process=not args.no_auto_process,
    )


if __name__ == "__main__":
    main()
