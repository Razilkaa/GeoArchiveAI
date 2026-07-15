from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class ArtifactRef:
    name: str
    path: str
    media_type: str = "application/json"
    role: str = "output"


@dataclass(slots=True)
class Issue:
    code: str
    message: str
    severity: str = "review"
    entity_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    name: str
    attributes: dict[str, Any]
    evidence: list[str]
    source_agent: str
    status: str = "extracted"


@dataclass(slots=True)
class AgentResult:
    agent: str
    status: AgentStatus
    started_at: str
    finished_at: str
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class AgentContext:
    report_id: str
    manifest_path: Path
    run_dir: Path
    workspace: Path
    manifest: dict[str, Any]
    privacy: dict[str, Any]
    state: dict[str, Any]

    @property
    def external_allowed(self) -> bool:
        return bool(self.privacy.get("external_processing"))


class AgentHandler(Protocol):
    def __call__(self, context: AgentContext) -> AgentResult: ...


def stable_entity_id(entity_type: str, name: str) -> str:
    normalized = name.casefold().replace("ё", "е")
    normalized = re.sub(r"[^0-9a-zа-я]+", "-", normalized).strip("-")
    return f"{entity_type}:{normalized or 'unnamed'}"


def atomic_json_write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path

