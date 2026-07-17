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


class MapControlPoint(BaseModel):
    pixel: tuple[float, float]
    map: tuple[float, float]
    name: str | None = None


class MapGeoreferenceRequest(BaseModel):
    target_crs: str
    control_points: list[MapControlPoint] = Field(min_length=3)
    cell_size: float | None = Field(default=None, gt=0)
    name: str = "digitized_surface"
