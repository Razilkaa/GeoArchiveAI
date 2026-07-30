from __future__ import annotations

import json
import mimetypes
import threading
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
from fastapi import HTTPException
from PIL import Image
from shapely.geometry import GeometryCollection, MultiPoint, Point

from app.config import settings
from app.services.reports import read_json, report_run_dir, resolved_artifact_path
from services.map_digitizer.georeference import export_raster_package
from services.map_digitizer.georeference_grid import (
    fit_similarity,
    georeference_grid,
)
from services.map_digitizer.survey_profiles import (
    find_survey_shape,
    profile_number_variants,
)


GEOREFERENCE_LOCK = threading.Lock()
RASTER_ARTIFACTS = (
    ("georef_raster", "raster", "Геопривязанный растр"),
    ("georef_world_file", "world_file", "World file"),
    ("georef_projection", "projection", "Система координат PRJ"),
    ("georef_footprint", "footprint", "Контур покрытия"),
    ("georef_metrics", "metrics", "Метрики геопривязки"),
    ("georef_package", "package", "Архив ArcGIS / Petrel"),
)
GRID_ARTIFACTS = (
    ("georef_cps3", "cps3", "Геопривязанный CPS-3"),
    ("georef_xyz", "xyz", "Геопривязанный XYZ"),
    ("georef_grid_projection", "prj", "Система координат Grid"),
)


def _safe_page_id(page_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in page_id)


def _profile_number_column(columns: Any) -> str:
    for column in columns:
        if str(column).strip().casefold() in {"n_prof", "n_profil", "nprof"}:
            return str(column)
    raise HTTPException(409, "profile_number_attribute_missing")


def _display_profile(profile_id: str) -> str:
    if len(profile_id) == 6 and profile_id.isdigit():
        return f"{profile_id[-2:]}-{profile_id[:4]}"
    return profile_id


def _page_context(report_id: str, page_id: str) -> tuple[Path, Path, Path, dict]:
    run_dir = report_run_dir(report_id)
    map_result_path = run_dir / "map_agent" / "result.json"
    map_result = read_json(map_result_path, {})
    job = next(
        (item for item in map_result.get("jobs", []) if item.get("page_id") == page_id),
        None,
    )
    if job is None:
        raise HTTPException(404, "report_map_page_not_found")
    job_dir = run_dir / "map_agent" / "jobs" / _safe_page_id(page_id)
    pipeline_result_path = job_dir / "pipeline_result.json"
    pipeline_result = read_json(pipeline_result_path, {})
    source = resolved_artifact_path(pipeline_result.get("source") or job.get("source"))
    if not source.is_file():
        raise HTTPException(409, "report_map_source_not_ready")
    return map_result_path, pipeline_result_path, source, map_result


def _inventory(report_id: str):
    shape_path = find_survey_shape(report_id, settings.project_root / "shapes")
    if shape_path is None:
        raise HTTPException(409, "survey_shape_not_found")
    frame = pyogrio.read_dataframe(shape_path, where=f"N_RGF = '{report_id}'")
    if not len(frame):
        raise HTTPException(409, "survey_profiles_not_found")
    if frame.crs is not None and frame.crs.is_geographic:
        target_crs = frame.estimate_utm_crs()
        if target_crs is not None:
            frame = frame.to_crs(target_crs)
    number_column = _profile_number_column(frame.columns)
    profiles = {}
    for _, row in frame.iterrows():
        profile_id = str(row[number_column]).strip()
        geometry = row.geometry
        if not profile_id or profile_id in {"nan", "None"} or geometry is None:
            continue
        if geometry.geom_type == "MultiLineString":
            geometry = max(geometry.geoms, key=lambda part: part.length)
        if geometry.geom_type != "LineString":
            continue
        existing = profiles.get(profile_id)
        if existing is None or geometry.length > existing.length:
            profiles[profile_id] = geometry
    return frame.crs, profiles


def _point_geometries(geometry: Any) -> list[Point]:
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for part in geometry.geoms if isinstance(part, Point)]
    return []


def _recognized_profile_ids(ocr_path: Path, available: set[str]) -> set[str]:
    payload = read_json(ocr_path, {})
    recognized = set()
    for item in payload.get("lines", []):
        if float(item.get("score") or 0.0) < 0.8:
            continue
        digits = "".join(character for character in str(item.get("text") or "") if character.isdigit())
        for candidate in profile_number_variants(digits):
            if candidate in available:
                recognized.add(candidate)
    return recognized


def _crossings(report_id: str, page_id: str) -> tuple[str, list[dict[str, Any]]]:
    _, pipeline_result_path, _, _ = _page_context(report_id, page_id)
    crs, profiles = _inventory(report_id)
    recognized = _recognized_profile_ids(
        pipeline_result_path.parent / "ocr.json",
        set(profiles),
    )
    crossings = []
    for left, right in combinations(sorted(recognized), 2):
        points = _point_geometries(profiles[left].intersection(profiles[right]))
        for index, point in enumerate(points):
            key = f"{left}:{right}:{index}"
            crossings.append(
                {
                    "key": key,
                    "profiles": [left, right],
                    "label": f"{_display_profile(left)} × {_display_profile(right)}",
                    "map": [float(point.x), float(point.y)],
                }
            )
    crossings.sort(key=lambda item: item["label"])
    return str(crs), crossings


def profile_crossings(report_id: str, page_id: str) -> dict[str, Any]:
    crs, crossings = _crossings(report_id, page_id)
    _, _, source, _ = _page_context(report_id, page_id)
    with Image.open(source) as image:
        raster_size = [int(image.width), int(image.height)]
    return {
        "report_id": report_id,
        "page_id": page_id,
        "target_crs": crs,
        "crossings": crossings,
        "raster_size": raster_size,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def georeference_report_map(
    report_id: str,
    *,
    page_id: str,
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    with GEOREFERENCE_LOCK:
        map_result_path, pipeline_result_path, source, map_result = _page_context(
            report_id,
            page_id,
        )
        target_crs, available_crossings = _crossings(report_id, page_id)
        crossing_by_key = {item["key"]: item for item in available_crossings}
        controls = []
        normalized_anchors = []
        with Image.open(source) as image:
            width, height = image.size
        for anchor in anchors:
            pixel = [float(value) for value in anchor["pixel"]]
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                raise HTTPException(422, "pixel_anchor_outside_raster")
            crossing_key = str(anchor["crossing_key"])
            crossing = crossing_by_key.get(crossing_key)
            if crossing is None:
                raise HTTPException(422, "profile_crossing_not_available")
            controls.append(
                {
                    "pixel": pixel,
                    "map": crossing["map"],
                    "name": crossing["label"],
                }
            )
            normalized_anchors.append(
                {
                    "pixel": pixel,
                    "crossing_key": crossing_key,
                    "profiles": crossing["profiles"],
                    "label": crossing["label"],
                    "map": crossing["map"],
                }
            )
        if len({item["crossing_key"] for item in normalized_anchors}) < 2:
            raise HTTPException(422, "two_distinct_profile_crossings_required")
        try:
            matrix, control_quality = fit_similarity(controls)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        output_dir = pipeline_result_path.parent / "georef_manual"
        raster_result = export_raster_package(
            source,
            output_dir,
            matrix,
            target_crs,
            {
                "method": "manual_two_profile_crossings",
                "quality_status": "anchored",
                "control_points": normalized_anchors,
                "control_quality": control_quality,
            },
        )
        pipeline_result = read_json(pipeline_result_path, {})
        files = raster_result["files"]
        pipeline_result.setdefault("artifacts", {}).update(
            {
                f"georef_{name}": path
                for name, path in files.items()
            }
        )
        pipeline_result.setdefault("stages", {})["georeference"] = {
            "status": "complete",
            "method": "manual_two_profile_crossings",
            "metrics": raster_result,
        }
        pipeline_result["status"] = "review"
        pipeline_result["quality"] = {
            "status": "review",
            "reasons": ["manual_two_point_georeference"],
        }

        grid_result = None
        pixel_grid = resolved_artifact_path(
            pipeline_result.get("artifacts", {}).get("pixel_grid")
        )
        if pixel_grid.is_file():
            grid_result = georeference_grid(
                pixel_grid,
                controls,
                output_dir / "grid",
                target_crs=target_crs,
                name="digitized_surface",
            )
            pipeline_result["artifacts"].update(
                {
                    "georef_cps3": grid_result["files"]["cps3"],
                    "georef_xyz": grid_result["files"]["xyz"],
                    "georef_grid_projection": grid_result["files"]["prj"],
                }
            )
        _write_json_atomic(pipeline_result_path, pipeline_result)

        generated_names = {
            *(item[0] for item in RASTER_ARTIFACTS),
            *(item[0] for item in GRID_ARTIFACTS),
        }
        artifacts = [
            item
            for item in map_result.get("artifacts", [])
            if not (
                item.get("page_id") == page_id
                and item.get("name") in generated_names
            )
        ]
        for artifact_name, file_key, label in RASTER_ARTIFACTS:
            path = files.get(file_key)
            if path:
                artifacts.append(
                    {
                        "name": artifact_name,
                        "label": f"{label} · {source.name}",
                        "path": path,
                        "media_type": mimetypes.guess_type(path)[0]
                        or "application/octet-stream",
                        "page_id": page_id,
                    }
                )
        if grid_result:
            for artifact_name, file_key, label in GRID_ARTIFACTS:
                path = grid_result["files"].get(file_key)
                if path:
                    artifacts.append(
                        {
                            "name": artifact_name,
                            "label": f"{label} · {source.name}",
                            "path": path,
                            "media_type": mimetypes.guess_type(path)[0]
                            or "application/octet-stream",
                            "page_id": page_id,
                        }
                    )
        for job in map_result.get("jobs", []):
            if job.get("page_id") == page_id:
                job["status"] = "review"
                job["quality"] = {
                    "status": "review",
                    "reasons": ["manual_two_point_georeference"],
                }
        map_result["artifacts"] = artifacts
        _write_json_atomic(map_result_path, map_result)
        return {
            "report_id": report_id,
            "page_id": page_id,
            "status": "anchored",
            "target_crs": target_crs,
            "control_points": normalized_anchors,
            "control_quality": control_quality,
            "raster": raster_result,
            "grid": grid_result,
        }
