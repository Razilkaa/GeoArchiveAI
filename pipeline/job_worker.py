from __future__ import annotations

import argparse
import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_contracts import atomic_json_write, utc_now
from build_result_bundle import build_bundle
from factory_agents import run_agents
from factory_runner import (
    build_provenance_markdown,
    ingest_ragflow,
    load_manifest,
    run_fast_ocr,
    run_api_ocr,
    run_vision_ocr,
)
from page_router import route_pages
from report_orchestrator import ReportOrchestrator


CORE_STAGES = ("page_routing", "fast_ocr", "markdown", "ragflow", "agents", "bundle", "operator")


@dataclass(slots=True)
class WorkerConfig:
    credentials_file: Path
    ragflow_token_file: Path
    dataset_id: str | None = None
    ragflow_base_url: str = "https://ragflow-dev.finam.ru/api/v1"
    proxy: str | None = "socks5h://127.0.0.1:7777"
    vision_model: str = "openai/gpt-4o-mini"
    agent_model: str = "openai/gpt-4o-mini"
    vision_workers: int = 4
    router_batch_size: int = 12
    paddleocr: Path | None = None
    ocr_api_url: str | None = "http://127.0.0.1:18080"
    allow_external: bool = True
    force: set[str] = field(default_factory=set)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_json(path: Path, predicate: Callable[[dict[str, Any]], bool]) -> bool:
    try:
        payload = _read_json(path, {})
        return isinstance(payload, dict) and predicate(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


@contextmanager
def report_lock(run_dir: Path, stale_after_s: int = 6 * 60 * 60):
    lock_path = run_dir / "worker" / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime > stale_after_s:
        lock_path.unlink()
    payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "created_at": utc_now()})
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Report is already being processed: {run_dir.name}") from error
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class JobWorker:
    def __init__(self, manifest_path: Path, config: WorkerConfig) -> None:
        self.manifest_path = manifest_path.resolve()
        self.run_dir = self.manifest_path.parent
        self.config = config
        self.manifest = load_manifest(self.manifest_path)
        self.report_id = str(self.manifest["report_id"])
        self.state_path = self.run_dir / "worker" / "state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        existing = _read_json(self.state_path)
        if isinstance(existing, dict) and existing.get("report_id") == self.report_id:
            existing.setdefault("stages", {})
            existing.setdefault("events", [])
            return existing
        return {
            "schema_version": 1,
            "report_id": self.report_id,
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "stages": {name: {"status": "pending", "attempts": 0} for name in CORE_STAGES},
            "events": [],
        }

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json_write(self.state_path, self.state)

    def _event(self, stage: str, event: str, **details: Any) -> None:
        self.state["events"].append({"at": utc_now(), "stage": stage, "event": event, **details})
        self.state["events"] = self.state["events"][-300:]
        self._save()

    def _external_allowed(self) -> bool:
        policy = _read_json(self.run_dir / "privacy.json", {}) or {}
        return self.config.allow_external or bool(policy.get("external_processing"))

    def _routing_valid(self) -> bool:
        manifest = load_manifest(self.manifest_path)
        return (
            manifest["stages"]["page_routing"]["status"] == "completed"
            and not manifest["queues"].get("route_vlm")
        )

    def _ocr_valid(self) -> bool:
        queue_path = self.run_dir / "fast_ocr" / "queue.json"
        output_dir = self.run_dir / "fast_ocr" / "output"
        try:
            queue = _read_json(queue_path, [])
            expected = {
                f"{Path(item['input_name']).stem}_res.json"
                for item in queue
            }
        except (KeyError, TypeError):
            return False
        return bool(expected) and all((output_dir / name).exists() for name in expected)

    def _markdown_path(self) -> Path:
        return self.run_dir / f"report_{self.report_id}_fast_ocr.md"

    def _markdown_valid(self) -> bool:
        path = self._markdown_path()
        return path.exists() and path.stat().st_size > 100

    def _ragflow_path(self) -> Path:
        return self.run_dir / "ragflow.json"

    def _ragflow_valid(self) -> bool:
        return _valid_json(
            self._ragflow_path(),
            lambda value: bool(value.get("document_id")) and (value.get("state") or {}).get("run") == "DONE",
        )

    def _agents_path(self) -> Path:
        return self.run_dir / "agent_results.json"

    def _agents_valid(self) -> bool:
        return _valid_json(
            self._agents_path(),
            lambda value: bool(value.get("agents"))
            and all(item.get("status") == "completed" for item in value["agents"].values()),
        )

    def _bundle_path(self) -> Path:
        return self.run_dir / "result_bundle.json"

    def _bundle_valid(self) -> bool:
        return _valid_json(
            self._bundle_path(),
            lambda value: value.get("report_id") == self.report_id
            and value.get("status") in {"completed", "review"},
        )

    def _dataset_id(self) -> str:
        if self.config.dataset_id:
            return self.config.dataset_id
        metadata = _read_json(self._ragflow_path(), {}) or {}
        dataset_id = metadata.get("dataset_id") or (metadata.get("state") or {}).get("dataset_id")
        if dataset_id:
            return str(dataset_id)
        raise ValueError("RAGFlow dataset id is required for a new report")

    def _run_stage(
        self,
        name: str,
        validator: Callable[[], bool],
        handler: Callable[[], Any],
        *,
        requires_external: bool = False,
    ) -> bool:
        record = self.state["stages"].setdefault(name, {"status": "pending", "attempts": 0})
        forced = name in self.config.force
        if not forced and validator():
            record.update({"status": "completed", "cached": True, "finished_at": utc_now()})
            record.pop("error", None)
            record.pop("detail", None)
            self._event(name, "cache_hit")
            return True
        if requires_external and not self._external_allowed():
            record.update(
                {
                    "status": "blocked",
                    "cached": False,
                    "finished_at": utc_now(),
                    "error": "external_processing_not_allowed",
                }
            )
            self._event(name, "blocked", reason="external_processing_not_allowed")
            return False
        record["status"] = "running"
        record["cached"] = False
        record["started_at"] = utc_now()
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record.pop("error", None)
        record.pop("detail", None)
        self._event(name, "started")
        started = time.perf_counter()
        try:
            output = handler()
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "elapsed_s": round(time.perf_counter() - started, 2),
                    "error": type(error).__name__,
                    "detail": str(error)[:1000],
                }
            )
            self._event(name, "failed", error=type(error).__name__)
            return False
        record.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "elapsed_s": round(time.perf_counter() - started, 2),
                "output": str(output) if isinstance(output, (str, Path)) else None,
            }
        )
        self._event(name, "completed")
        return True

    def run(self) -> dict[str, Any]:
        with report_lock(self.run_dir):
            self.state["status"] = "running"
            self.state["started_at"] = utc_now()
            self._save()

            routing_external = bool(load_manifest(self.manifest_path)["queues"].get("route_vlm"))
            stages = (
                (
                    "page_routing",
                    self._routing_valid,
                    lambda: route_pages(
                        self.manifest_path,
                        self.config.credentials_file,
                        model=self.config.vision_model,
                        batch_size=self.config.router_batch_size,
                        workers=self.config.vision_workers,
                    ),
                    routing_external,
                ),
                (
                    "fast_ocr",
                    self._ocr_valid,
                    lambda: run_api_ocr(
                        self.manifest_path, self.config.ocr_api_url, self.config.vision_workers
                    )
                    if self.config.ocr_api_url
                    else (
                        run_fast_ocr(self.manifest_path, self.config.paddleocr, self.config.vision_workers)
                        if self.config.paddleocr
                        else run_vision_ocr(
                            self.manifest_path,
                            self.config.credentials_file,
                            self.config.vision_model,
                            self.config.vision_workers,
                        )
                    ),
                    self.config.ocr_api_url is None and self.config.paddleocr is None,
                ),
                ("markdown", self._markdown_valid, lambda: build_provenance_markdown(self.manifest_path), False),
                (
                    "ragflow",
                    self._ragflow_valid,
                    lambda: ingest_ragflow(
                        self.manifest_path,
                        self._markdown_path(),
                        self._dataset_id(),
                        self.config.ragflow_token_file,
                        self.config.ragflow_base_url,
                        self.config.proxy,
                    ),
                    False,
                ),
                (
                    "agents",
                    self._agents_valid,
                    lambda: run_agents(
                        self.manifest_path,
                        self._ragflow_path(),
                        self.config.ragflow_token_file,
                        self.config.credentials_file,
                        self.config.agent_model,
                        self.config.ragflow_base_url,
                        self.config.proxy,
                    ),
                    True,
                ),
                (
                    "bundle",
                    self._bundle_valid,
                    lambda: build_bundle(
                        self.manifest_path,
                        self._ragflow_path(),
                        self._agents_path(),
                        self.run_dir / "map_agent" / "result.json",
                    ),
                    False,
                ),
                (
                    "operator",
                    lambda: False,
                    lambda: ReportOrchestrator(self.manifest_path, workers=2).run(),
                    False,
                ),
            )
            for name, validator, handler, external in stages:
                if not self._run_stage(name, validator, handler, requires_external=external):
                    break

            statuses = [item["status"] for item in self.state["stages"].values()]
            if "failed" in statuses:
                self.state["status"] = "failed"
            elif "blocked" in statuses:
                self.state["status"] = "blocked"
            elif all(self.state["stages"].get(name, {}).get("status") == "completed" for name in CORE_STAGES):
                self.state["status"] = "completed"
            else:
                self.state["status"] = "partial"
            self.state["finished_at"] = utc_now()
            self._save()
            return self.state


def discover_dataset_id(runs_root: Path) -> str | None:
    for path in sorted(runs_root.glob("*/ragflow.json")):
        payload = _read_json(path, {}) or {}
        dataset_id = payload.get("dataset_id") or (payload.get("state") or {}).get("dataset_id")
        if dataset_id:
            return str(dataset_id)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable end-to-end report jobs")
    parser.add_argument("report_ids", nargs="*")
    parser.add_argument("--all", action="store_true", dest="all_reports")
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).parents[1] / "runs")
    parser.add_argument("--credentials-file", type=Path, default=Path(__file__).parents[1] / "secrets" / "tokent.txt")
    parser.add_argument("--token-file", type=Path, default=Path(__file__).parents[1] / "secrets" / "ragflow_token.txt")
    parser.add_argument("--dataset-id")
    parser.add_argument("--ragflow-base-url", default="https://ragflow-dev.finam.ru/api/v1")
    parser.add_argument("--proxy", default="socks5h://127.0.0.1:7777")
    parser.add_argument("--vision-model", default="openai/gpt-4o-mini")
    parser.add_argument("--agent-model", default="openai/gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--router-batch-size", type=int, default=12)
    parser.add_argument("--paddleocr", type=Path)
    parser.add_argument("--ocr-api-url", default="http://127.0.0.1:18080")
    parser.add_argument("--force", action="append", choices=CORE_STAGES, default=[])
    args = parser.parse_args()

    if args.all_reports:
        manifests = sorted(args.runs_root.glob("*/job.json"))
    else:
        if not args.report_ids:
            parser.error("provide report_ids or --all")
        manifests = [args.runs_root / report_id / "job.json" for report_id in args.report_ids]
    dataset_id = args.dataset_id or discover_dataset_id(args.runs_root)
    results = []
    for manifest in manifests:
        if not manifest.exists():
            results.append({"report_id": manifest.parent.name, "status": "missing_manifest"})
            continue
        config = WorkerConfig(
            credentials_file=args.credentials_file,
            ragflow_token_file=args.token_file,
            dataset_id=dataset_id,
            ragflow_base_url=args.ragflow_base_url,
            proxy=args.proxy or None,
            vision_model=args.vision_model,
            agent_model=args.agent_model,
            vision_workers=args.workers,
            router_batch_size=args.router_batch_size,
            paddleocr=args.paddleocr,
            ocr_api_url=args.ocr_api_url or None,
            force=set(args.force),
        )
        state = JobWorker(manifest, config).run()
        results.append({"report_id": state["report_id"], "status": state["status"], "state": str(manifest.parent / "worker" / "state.json")})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
