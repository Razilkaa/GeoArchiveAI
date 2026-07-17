from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.schemas import MapGeoreferenceRequest
from app.services.map_jobs import (
    georeference_map_job as run_georeference,
    map_job_artifact,
    map_job_status,
    start_map_job,
)


router = APIRouter(prefix="/api/maps", tags=["maps"])


@router.post("/jobs", status_code=202)
async def create_map_job(
    file: UploadFile = File(...),
    interval: float | None = None,
    trace_scale: float = 0.6,
) -> dict:
    return await run_in_threadpool(
        start_map_job,
        file.filename or "map.jpg",
        file.file,
        interval=interval,
        trace_scale=trace_scale,
    )


@router.get("/jobs/{job_id}")
def get_map_job(job_id: str) -> dict:
    return map_job_status(job_id)


@router.get("/jobs/{job_id}/artifacts/{artifact_name}")
def download_map_artifact(job_id: str, artifact_name: str) -> FileResponse:
    path = map_job_artifact(job_id, artifact_name)
    media_types = {
        ".png": "image/png",
        ".geojson": "application/geo+json",
        ".json": "application/json",
        ".cps3": "text/plain",
        ".xyz": "text/plain",
        ".prj": "text/plain",
        ".npz": "application/octet-stream",
    }
    return FileResponse(path, media_type=media_types.get(path.suffix.lower()))


@router.post("/jobs/{job_id}/georeference")
async def georeference_map_job(job_id: str, request: MapGeoreferenceRequest) -> dict:
    controls = [point.model_dump() for point in request.control_points]
    return await run_in_threadpool(
        run_georeference,
        job_id,
        target_crs=request.target_crs,
        control_points=controls,
        cell_size=request.cell_size,
        name=request.name,
    )
