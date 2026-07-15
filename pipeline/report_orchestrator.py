from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from agent_contracts import (
    AgentContext,
    AgentHandler,
    AgentResult,
    AgentStatus,
    ArtifactRef,
    EntityRecord,
    Issue,
    atomic_json_write,
    stable_entity_id,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class AgentNode:
    name: str
    handler: AgentHandler
    requires: tuple[str, ...] = ()
    observes: tuple[str, ...] = ()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_exists(record: dict[str, Any]) -> bool:
    artifacts = record.get("result", {}).get("artifacts", [])
    return bool(artifacts) and all(Path(item["path"]).exists() for item in artifacts)


def make_result(
    agent: str,
    status: AgentStatus,
    started_at: str,
    *,
    artifacts: list[ArtifactRef] | None = None,
    metrics: dict[str, Any] | None = None,
    issues: list[Issue] | None = None,
    message: str | None = None,
) -> AgentResult:
    return AgentResult(
        agent=agent,
        status=status,
        started_at=started_at,
        finished_at=utc_now(),
        artifacts=artifacts or [],
        metrics=metrics or {},
        issues=issues or [],
        message=message,
    )


def classifier_agent(context: AgentContext) -> AgentResult:
    started = utc_now()
    pages = context.manifest.get("pages", [])
    role_counts: dict[str, int] = {}
    routes: dict[str, list[str]] = {"rag": [], "map": [], "review": []}
    for page in pages:
        role = str(page.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        if role in {"text", "toc"}:
            routes["rag"].append(page["id"])
        if role in {"graphic", "sample"}:
            routes["map"].append(page["id"])
        if role == "unknown":
            routes["review"].append(page["id"])
    payload = {
        "schema_version": 1,
        "report_id": context.report_id,
        "generated_at": utc_now(),
        "source_root": context.manifest["source_root"],
        "role_counts": dict(sorted(role_counts.items())),
        "routes": routes,
    }
    output = atomic_json_write(context.workspace / "classification.json", payload)
    issues = []
    if routes["review"]:
        issues.append(
            Issue(
                code="unclassified_pages",
                message=f"Требуют маршрутизации: {len(routes['review'])} стр.",
                severity="review",
            )
        )
    return make_result(
        "classifier",
        AgentStatus.COMPLETED,
        started,
        artifacts=[ArtifactRef("classification", str(output))],
        metrics={"pages": len(pages), "roles": role_counts, "review_pages": len(routes["review"])},
        issues=issues,
    )


def _entity_name(entity_type: str, item: dict[str, Any], index: int) -> str:
    for key in ("name", "title", "topic", "map_title", "well"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    statement = str(item.get("statement") or "").strip()
    return statement[:90] if statement else f"{entity_type}-{index + 1}"


def _normalize_rag_entities(source: dict[str, Any]) -> list[dict[str, Any]]:
    if "entities" in source:
        groups = source.get("entities", {})
    else:
        groups = {
            name: result.get("result", {}).get("items", [])
            for name, result in source.get("agents", {}).items()
            if result.get("status") == "completed"
        }
    type_names = {
        "structures": "structure",
        "wells": "well",
        "maps": "map",
        "horizons": "horizon",
        "key_results": "finding",
    }
    records: dict[str, EntityRecord] = {}
    for group_name, items in groups.items():
        entity_type = type_names.get(group_name, group_name.rstrip("s") or "entity")
        for index, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            name = _entity_name(entity_type, item, index)
            entity_id = stable_entity_id(entity_type, name)
            evidence = sorted({str(value) for value in item.get("evidence", []) if value})
            attributes = {key: value for key, value in item.items() if key not in {"name", "evidence"}}
            if entity_id in records:
                current = records[entity_id]
                current.evidence = sorted(set(current.evidence) | set(evidence))
                current.attributes.setdefault("observations", []).append(attributes)
                continue
            records[entity_id] = EntityRecord(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                attributes=attributes,
                evidence=evidence,
                source_agent="rag",
                status="extracted" if evidence else "review",
            )
    return [
        {
            "entity_id": item.entity_id,
            "entity_type": item.entity_type,
            "name": item.name,
            "attributes": item.attributes,
            "evidence": item.evidence,
            "source_agent": item.source_agent,
            "status": item.status,
        }
        for item in records.values()
    ]


def rag_agent(context: AgentContext) -> AgentResult:
    started = utc_now()
    candidates = (
        context.run_dir / "result_bundle.json",
        context.run_dir / "agent_results_fast.json",
        context.run_dir / "agent_results.json",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        return make_result(
            "rag",
            AgentStatus.BLOCKED,
            started,
            issues=[
                Issue(
                    code="rag_artifact_missing",
                    message="Сначала требуется OCR и извлечение сущностей.",
                    severity="blocking",
                )
            ],
            message="Нет готового результата RAG-ветки.",
        )
    entities = _normalize_rag_entities(load_json(source_path, {}))
    output = atomic_json_write(
        context.workspace / "rag_entities.json",
        {
            "schema_version": 1,
            "report_id": context.report_id,
            "generated_at": utc_now(),
            "source": str(source_path),
            "entities": entities,
        },
    )
    missing_evidence = [item["entity_id"] for item in entities if not item["evidence"]]
    issues = []
    if missing_evidence:
        issues.append(
            Issue(
                code="entities_without_evidence",
                message=f"Сущности без источника: {len(missing_evidence)}.",
                entity_ids=missing_evidence,
            )
        )
    counts: dict[str, int] = {}
    for entity in entities:
        counts[entity["entity_type"]] = counts.get(entity["entity_type"], 0) + 1
    return make_result(
        "rag",
        AgentStatus.COMPLETED,
        started,
        artifacts=[ArtifactRef("rag_entities", str(output))],
        metrics={"entities": len(entities), "types": counts, "source": source_path.name},
        issues=issues,
    )


def map_agent(context: AgentContext) -> AgentResult:
    started = utc_now()
    result_path = context.run_dir / "map_agent" / "result.json"
    if not result_path.exists():
        return make_result(
            "map",
            AgentStatus.BLOCKED,
            started,
            issues=[
                Issue(
                    code="map_adapter_waiting",
                    message="Ожидается результат выбранного open-source digitizer.",
                    severity="blocking",
                )
            ],
            message=f"Адаптер должен записать результат в {result_path}",
        )
    payload = load_json(result_path, {})
    if payload.get("status") != "completed":
        return make_result(
            "map",
            AgentStatus.BLOCKED,
            started,
            issues=[Issue("map_result_incomplete", "Картографический результат ещё не завершён.", "blocking")],
        )
    artifacts = []
    for item in payload.get("artifacts", []):
        path = Path(str(item.get("path") or ""))
        if path.exists():
            artifacts.append(
                ArtifactRef(
                    name=str(item.get("name") or path.stem),
                    path=str(path),
                    media_type=str(item.get("media_type") or "application/octet-stream"),
                )
            )
    if not artifacts:
        return make_result(
            "map",
            AgentStatus.BLOCKED,
            started,
            issues=[Issue("map_artifacts_missing", "Картограф не передал существующие артефакты.", "blocking")],
        )
    return make_result(
        "map",
        AgentStatus.COMPLETED,
        started,
        artifacts=artifacts,
        metrics=payload.get("metrics", {}),
        issues=[Issue(**item) for item in payload.get("issues", [])],
    )


def _canonical_name(value: str) -> str:
    normalized = value.casefold().replace("ё", "е")
    for suffix in ("структура", "площадь", "скважина"):
        normalized = normalized.replace(suffix, "")
    return " ".join("".join(character if character.isalnum() else " " for character in normalized).split())


def qc_agent(context: AgentContext) -> AgentResult:
    started = utc_now()
    rag_payload = load_json(context.workspace / "rag_entities.json", {})
    entities = rag_payload.get("entities", [])
    issues: list[Issue] = []
    for entity in entities:
        if not entity.get("evidence"):
            issues.append(
                Issue(
                    code="missing_evidence",
                    message=f"Нет источника: {entity.get('name')}",
                    entity_ids=[entity["entity_id"]],
                )
            )
    for index, first in enumerate(entities):
        if first.get("entity_type") not in {"structure", "well"}:
            continue
        first_name = _canonical_name(str(first.get("name") or ""))
        if len(first_name) < 5:
            continue
        for second in entities[index + 1 :]:
            if first.get("entity_type") != second.get("entity_type"):
                continue
            second_name = _canonical_name(str(second.get("name") or ""))
            ratio = SequenceMatcher(None, first_name, second_name).ratio()
            if 0.78 <= ratio < 1.0:
                issues.append(
                    Issue(
                        code="possible_alias",
                        message=f"Возможные варианты одного названия: {first['name']} / {second['name']}",
                        entity_ids=[first["entity_id"], second["entity_id"]],
                    )
                )
    map_state = context.state.get("agents", {}).get("map", {})
    map_available = map_state.get("status") == AgentStatus.COMPLETED.value
    if not map_available:
        issues.append(
            Issue(
                code="map_crosscheck_pending",
                message="Перекрёстная проверка по карте ожидает картографический агент.",
                severity="info",
            )
        )
    elif map_state.get("result", {}).get("metrics", {}).get("quality_status") == "review":
        issues.append(
            Issue(
                code="map_quality_review",
                message="Картографический слой создан, но требует экспертной проверки.",
                severity="review",
            )
        )
    registry = []
    for entity in entities:
        item = dict(entity)
        item["status"] = "confirmed_text" if entity.get("evidence") else "review"
        item["checks"] = ["evidence"] if entity.get("evidence") else []
        registry.append(item)
    registry_path = atomic_json_write(
        context.workspace / "entity_registry.json",
        {
            "schema_version": 1,
            "report_id": context.report_id,
            "generated_at": utc_now(),
            "entities": registry,
        },
    )
    qc_path = atomic_json_write(
        context.workspace / "qc_report.json",
        {
            "schema_version": 1,
            "report_id": context.report_id,
            "generated_at": utc_now(),
            "status": "review" if any(item.severity != "info" for item in issues) else "pass",
            "map_crosscheck": "completed" if map_available else "pending",
            "issues": [
                {
                    "code": item.code,
                    "message": item.message,
                    "severity": item.severity,
                    "entity_ids": item.entity_ids,
                    "evidence": item.evidence,
                }
                for item in issues
            ],
        },
    )
    return make_result(
        "qc",
        AgentStatus.COMPLETED,
        started,
        artifacts=[
            ArtifactRef("entity_registry", str(registry_path)),
            ArtifactRef("qc_report", str(qc_path)),
        ],
        metrics={
            "entities": len(registry),
            "issues": len(issues),
            "map_crosscheck": "completed" if map_available else "pending",
        },
        issues=issues,
    )


def export_agent(context: AgentContext) -> AgentResult:
    started = utc_now()
    registry = load_json(context.workspace / "entity_registry.json", {})
    qc = load_json(context.workspace / "qc_report.json", {})
    map_state = context.state.get("agents", {}).get("map", {})
    map_completed = map_state.get("status") == AgentStatus.COMPLETED.value
    map_quality = map_state.get("result", {}).get("metrics", {}).get("quality_status")
    export_status = "completed" if map_completed and map_quality != "review" else "review"
    payload = {
        "schema_version": 1,
        "report_id": context.report_id,
        "generated_at": utc_now(),
        "status": export_status,
        "source_root": context.manifest["source_root"],
        "entities": registry.get("entities", []),
        "qc": qc,
        "map": map_state.get("result"),
        "provenance": {
            "manifest": str(context.manifest_path),
            "privacy": context.privacy,
        },
    }
    output = atomic_json_write(context.workspace / "report_result.json", payload)
    return make_result(
        "export",
        AgentStatus.COMPLETED,
        started,
        artifacts=[ArtifactRef("report_result", str(output))],
        metrics={"status": payload["status"], "entities": len(payload["entities"])},
        issues=(
            [
                Issue(
                    "map_export_review",
                    "Геопространственный экспорт требует проверки картографического слоя.",
                    "review" if map_completed else "info",
                )
            ]
            if payload["status"] == "review"
            else []
        ),
    )


class ReportOrchestrator:
    def __init__(self, manifest_path: Path, workers: int = 2):
        self.manifest_path = manifest_path.resolve()
        self.run_dir = self.manifest_path.parent
        self.workspace = self.run_dir / "orchestrator"
        self.state_path = self.workspace / "state.json"
        self.workers = max(1, workers)
        self.manifest = load_json(self.manifest_path)
        if not self.manifest:
            raise ValueError(f"Manifest not found: {self.manifest_path}")
        self.privacy = load_json(
            self.run_dir / "privacy.json",
            {
                "classification": "existing_demo",
                "external_processing": False,
                "note": "The orchestrator itself performs local artifact adoption only.",
            },
        )
        self.nodes = {
            node.name: node
            for node in (
                AgentNode("classifier", classifier_agent),
                AgentNode("rag", rag_agent, requires=("classifier",)),
                AgentNode("map", map_agent, requires=("classifier",)),
                AgentNode("qc", qc_agent, requires=("classifier", "rag"), observes=("map",)),
                AgentNode("export", export_agent, requires=("qc",), observes=("map",)),
            )
        }
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        existing = load_json(self.state_path)
        if existing:
            for name, node in self.nodes.items():
                existing.setdefault("agents", {}).setdefault(
                    name,
                    {"status": AgentStatus.PENDING.value, "requires": list(node.requires), "observes": list(node.observes)},
                )
            return existing
        state = {
            "schema_version": 1,
            "report_id": self.manifest["report_id"],
            "manifest": str(self.manifest_path),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "pending",
            "policy": {
                "classification": self.privacy.get("classification", "unknown"),
                "external_processing": bool(self.privacy.get("external_processing")),
            },
            "agents": {
                name: {
                    "status": AgentStatus.PENDING.value,
                    "requires": list(node.requires),
                    "observes": list(node.observes),
                }
                for name, node in self.nodes.items()
            },
            "events": [],
        }
        atomic_json_write(self.state_path, state)
        return state

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json_write(self.state_path, self.state)

    def _context(self) -> AgentContext:
        return AgentContext(
            report_id=self.manifest["report_id"],
            manifest_path=self.manifest_path,
            run_dir=self.run_dir,
            workspace=self.workspace,
            manifest=self.manifest,
            privacy=self.privacy,
            state=self.state,
        )

    def plan(self) -> dict[str, Any]:
        return {
            "report_id": self.manifest["report_id"],
            "policy": self.state["policy"],
            "agents": [
                {
                    "name": name,
                    "status": self.state["agents"][name]["status"],
                    "requires": list(node.requires),
                    "observes": list(node.observes),
                }
                for name, node in self.nodes.items()
            ],
        }

    def _execute(self, name: str) -> AgentResult:
        started = time.perf_counter()
        try:
            result = self.nodes[name].handler(self._context())
            result.metrics.setdefault("elapsed_s", round(time.perf_counter() - started, 3))
            return result
        except Exception as error:
            return make_result(
                name,
                AgentStatus.FAILED,
                utc_now(),
                issues=[Issue("agent_exception", type(error).__name__, "blocking")],
                message=str(error)[:1000],
            )

    def _invalidate_downstream(self, seeds: set[str]) -> None:
        affected = set(seeds)
        changed = True
        while changed:
            changed = False
            for name, node in self.nodes.items():
                if name in affected:
                    continue
                if any(dependency in affected for dependency in (*node.requires, *node.observes)):
                    affected.add(name)
                    changed = True
        for name in affected - seeds:
            record = self.state["agents"][name]
            record["status"] = AgentStatus.PENDING.value
            record.pop("result", None)

    def run(self, force: set[str] | None = None) -> dict[str, Any]:
        force = force or set()
        unknown = force - set(self.nodes)
        if unknown:
            raise ValueError(f"Unknown agents: {', '.join(sorted(unknown))}")
        retry_statuses: dict[str, str] = {}
        invalidation_seeds = set(force)
        for name, record in self.state["agents"].items():
            previous_status = record.get("status")
            if name in force or previous_status == AgentStatus.BLOCKED.value:
                retry_statuses[name] = previous_status
                record["status"] = AgentStatus.PENDING.value
                record.pop("result", None)
            elif previous_status == AgentStatus.COMPLETED.value and not artifact_exists(record):
                invalidation_seeds.add(name)
                record["status"] = AgentStatus.PENDING.value
                record.pop("result", None)
        if invalidation_seeds:
            self._invalidate_downstream(invalidation_seeds)
        self.state["status"] = "running"
        self._save()

        while True:
            ready = []
            for name, node in self.nodes.items():
                record = self.state["agents"][name]
                if record["status"] != AgentStatus.PENDING.value:
                    continue
                required_statuses = [self.state["agents"][dependency]["status"] for dependency in node.requires]
                observed_statuses = [self.state["agents"][dependency]["status"] for dependency in node.observes]
                observed_terminal = all(
                    status
                    in {
                        AgentStatus.COMPLETED.value,
                        AgentStatus.BLOCKED.value,
                        AgentStatus.SKIPPED.value,
                        AgentStatus.FAILED.value,
                    }
                    for status in observed_statuses
                )
                if all(
                    status in {AgentStatus.COMPLETED.value, AgentStatus.SKIPPED.value}
                    for status in required_statuses
                ) and observed_terminal:
                    ready.append(name)
                elif any(status == AgentStatus.FAILED.value for status in required_statuses):
                    record["status"] = AgentStatus.BLOCKED.value
                    record["result"] = make_result(
                        name,
                        AgentStatus.BLOCKED,
                        utc_now(),
                        message="Обязательный агент завершился ошибкой.",
                    ).to_dict()
                    self._save()
            if not ready:
                break
            for name in ready:
                self.state["agents"][name]["status"] = AgentStatus.RUNNING.value
                self.state["events"].append({"at": utc_now(), "agent": name, "event": "started"})
            self._save()
            with ThreadPoolExecutor(max_workers=min(self.workers, len(ready))) as pool:
                futures = {pool.submit(self._execute, name): name for name in ready}
                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    self.state["agents"][name]["status"] = result.status.value
                    self.state["agents"][name]["result"] = result.to_dict()
                    if (
                        retry_statuses.get(name) == AgentStatus.BLOCKED.value
                        and result.status == AgentStatus.COMPLETED
                    ):
                        self._invalidate_downstream({name})
                    self.state["events"].append(
                        {"at": utc_now(), "agent": name, "event": result.status.value}
                    )
                    self._save()

        statuses = [record["status"] for record in self.state["agents"].values()]
        if AgentStatus.FAILED.value in statuses:
            self.state["status"] = "failed"
        elif AgentStatus.PENDING.value in statuses or AgentStatus.RUNNING.value in statuses:
            self.state["status"] = "waiting"
        elif AgentStatus.BLOCKED.value in statuses:
            self.state["status"] = "partial"
        else:
            self.state["status"] = "completed"
        self._save()
        return self.state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumable multi-agent report operator")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("command", choices=("plan", "run", "status"), nargs="?", default="run")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="append", default=[])
    args = parser.parse_args()
    operator = ReportOrchestrator(args.manifest, workers=args.workers)
    if args.command == "plan":
        payload = operator.plan()
    elif args.command == "status":
        payload = operator.state
    else:
        payload = operator.run(set(args.force))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
