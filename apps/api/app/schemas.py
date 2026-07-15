from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    mode: Literal["live", "saved", "retrieval_only"] = "live"


class RunRequest(BaseModel):
    force: list[str] = Field(default_factory=list)


class IntakeResponse(BaseModel):
    status: str
    registered: int
    report_ids: list[str]
    unsupported: list[str]
    uploaded_name: str | None = None
    started: int = 0
    queued: int = 0
