from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import settings
from app.services.intake import register_inbox
from app.services.jobs import start_report
from app.services.reports import list_reports


_reconcile_lock = Lock()


def inbox_fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    records: list[tuple[str, int, int]] = []
    if not root.exists():
        return ()
    for path in root.rglob("*"):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        try:
            stat = path.stat()
            records.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        except OSError:
            return (("<changing>", 0, time.time_ns()),)
    return tuple(sorted(records))


def start_pending_reports(max_reports: int | None = None) -> dict[str, Any]:
    limit = max(1, max_reports or settings.auto_max_reports)
    reports = list_reports()
    active = sum(item["worker_status"] == "running" for item in reports)
    started: list[str] = []
    queued: list[str] = []
    for item in reports:
        if item["worker_status"] != "pending":
            continue
        report_id = str(item["report_id"])
        if active >= limit:
            queued.append(report_id)
            continue
        result = start_report(report_id, [])
        if result["status"] == "accepted":
            started.append(report_id)
            active += 1
        else:
            queued.append(report_id)
    return {"started": started, "queued": queued, "active": active}


def queue_status() -> dict[str, Any]:
    reports = list_reports()
    counts = {"running": 0, "pending": 0, "completed": 0, "failed": 0, "blocked": 0}
    for report in reports:
        status = str(report["worker_status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "enabled": settings.auto_intake_enabled,
        "interval_s": settings.auto_intake_interval_s,
        "settle_s": settings.auto_intake_settle_s,
        "max_parallel_reports": settings.auto_max_reports,
        "counts": counts,
    }


def reconcile_now(register: bool = True) -> dict[str, Any]:
    if not _reconcile_lock.acquire(blocking=False):
        return {"status": "busy", "registered": [], "started": [], "queued": []}
    try:
        summary = register_inbox() if register else {"registered": [], "unsupported": []}
        queue = start_pending_reports()
        return {
            "status": "completed",
            "registered": summary.get("registered", []),
            "unsupported": summary.get("unsupported", []),
            **queue,
        }
    finally:
        _reconcile_lock.release()


@dataclass
class InboxMonitor:
    observed: tuple[tuple[str, int, int], ...] | None = None
    observed_since: float = field(default_factory=time.monotonic)
    processed: tuple[tuple[str, int, int], ...] | None = None

    def tick(self) -> dict[str, Any]:
        current = inbox_fingerprint(settings.inbox_root)
        now = time.monotonic()
        if current != self.observed:
            self.observed = current
            self.observed_since = now
            return reconcile_now(register=False)
        stable = now - self.observed_since >= max(1, settings.auto_intake_settle_s)
        should_register = stable and current != self.processed
        result = reconcile_now(register=should_register)
        if should_register and result["status"] == "completed":
            self.processed = current
        return result

    async def run(self) -> None:
        while True:
            await asyncio.to_thread(self.tick)
            await asyncio.sleep(max(2, settings.auto_intake_interval_s))


async def stop_monitor(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
