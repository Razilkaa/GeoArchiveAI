"""End-to-end deterministic map digitization pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import time
from pathlib import Path
from typing import Callable

import requests

from services.map_digitizer.assign_contour_values import run as assign_values
from services.map_digitizer.depth_mark_surface import run as build_depth_surface
from services.map_digitizer.reconstruct_traced_surface import run as reconstruct_surface
from services.map_digitizer.trace_guided_surface import run as reconstruct_trace_guided
from services.map_digitizer.trace_map_isolines import trace


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_ocr(image_path: Path, api_url: str, timeout: float = 120.0) -> dict:
    session = requests.Session()
    session.trust_env = False
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.open("rb") as source:
        response = session.post(
            f"{api_url.rstrip('/')}/v1/ocr",
            files={"file": (image_path.name, source, mime_type)},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("lines"), list):
        raise ValueError("OCR response has no lines array")
    return payload


def quality_decision(assignment: dict, reconstruction: dict) -> dict:
    interval_m = float(assignment["contour_interval_km"]) * 1000.0
    p95_error = float(reconstruction["grid_quality"]["constraint_p95_abs_error_m"])
    reasons = []
    if assignment["confident_polylines"] < 3:
        reasons.append("fewer_than_3_direct_contours")
    if reconstruction["topology"]["crossing_pairs"]:
        reasons.append("contour_crossings")
    topology = reconstruction["topology"]
    if int(topology.get("closed_segments", 0)) > max(
        8, int(topology.get("levels", 0)) * 2
    ):
        reasons.append("excessive_closed_contours")
    if p95_error > interval_m * 0.5:
        reasons.append("constraint_error_exceeds_half_interval")
    if not reconstruction["grid_quality"]["value_range_preserved"]:
        reasons.append("value_range_not_preserved")
    inference = assignment.get("contour_interval_inference")
    if inference and float(inference.get("selected_support", 0.0)) < 0.7:
        reasons.append("low_contour_interval_confidence")
    return {
        "status": "accepted" if not reasons else "review",
        "reasons": reasons,
        "constraint_p95_fraction_of_interval": round(p95_error / interval_m, 4),
    }


def select_reconstruction_mode(assignment: dict) -> str:
    return (
        "dense_profile_measurements"
        if int(assignment.get("profile_measurements", 0)) >= 20
        else "sparse_labels_trace_guided"
    )


def reconstruct_adaptive(
    assignment: dict,
    assignment_path: Path,
    traces_path: Path,
    ocr_path: Path,
    image_path: Path,
    output_dir: Path,
) -> dict:
    interval = float(assignment["contour_interval_km"])
    mode = select_reconstruction_mode(assignment)
    if mode == "dense_profile_measurements":
        metrics = reconstruct_surface(
            assignment_path, image_path, output_dir, interval=interval, iterations=3
        )
        metrics["reconstruction_mode"] = mode
        return metrics

    preliminary_dir = output_dir / "preliminary"
    preliminary = build_depth_surface(
        ocr_path, image_path, preliminary_dir, interval=interval
    )
    metrics = reconstruct_trace_guided(
        traces_path,
        Path(preliminary["files"]["surface"]),
        output_dir,
        interval=interval,
        trace_image_path=image_path,
        label_paths=[ocr_path],
    )
    metrics["reconstruction_mode"] = mode
    metrics["preliminary_surface"] = preliminary
    return metrics


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    ocr_api_url: str = "http://127.0.0.1:18080",
    interval: float | None = None,
    trace_scale: float = 0.6,
    ocr_timeout: float = 120.0,
    ocr_client: Callable[[Path, str, float], dict] = request_ocr,
) -> dict:
    image_path = image_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "pipeline_result.json"
    result = {
        "version": 1,
        "source": str(image_path),
        "source_sha256": file_sha256(image_path),
        "status": "running",
        "stages": {},
        "artifacts": {},
    }

    def execute(name: str, operation):
        started = time.perf_counter()
        try:
            value = operation()
        except Exception as error:
            result["status"] = "failed"
            result["failed_stage"] = name
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            result["stages"][name] = {
                "status": "failed",
                "latency_s": round(time.perf_counter() - started, 3),
            }
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        result["stages"][name] = {
            "status": "complete",
            "latency_s": round(time.perf_counter() - started, 3),
        }
        return value

    ocr_path = output_dir / "ocr.json"
    if ocr_path.exists():
        ocr_payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        result["stages"]["ocr"] = {"status": "cached", "latency_s": 0.0}
    else:
        ocr_payload = execute(
            "ocr", lambda: ocr_client(image_path, ocr_api_url, ocr_timeout)
        )
        ocr_path.write_text(json.dumps(ocr_payload, ensure_ascii=False), encoding="utf-8")
    result["artifacts"]["ocr"] = str(ocr_path)

    trace_dir = output_dir / "trace"
    summary_path = execute(
        "trace",
        lambda: trace(image_path, None, trace_dir, trace_scale, ocr_path),
    )
    trace_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    result["stages"]["trace"]["metrics"] = trace_metrics

    assignment_dir = output_dir / "assignment"
    assignment = execute(
        "assignment",
        lambda: assign_values(
            trace_dir / "isolines.json",
            ocr_path,
            image_path,
            assignment_dir,
            interval=interval,
        ),
    )
    result["stages"]["assignment"]["metrics"] = assignment

    reconstruction_dir = output_dir / "surface"
    reconstruction = execute(
        "reconstruction",
        lambda: reconstruct_adaptive(
            assignment,
            assignment_dir / "valued_contours_pixels.geojson",
            trace_dir / "isolines.json",
            ocr_path,
            image_path,
            reconstruction_dir,
        ),
    )
    result["stages"]["reconstruction"]["metrics"] = reconstruction
    result["quality"] = quality_decision(assignment, reconstruction)
    result["status"] = result["quality"]["status"]
    result["total_latency_s"] = round(
        sum(float(stage["latency_s"]) for stage in result["stages"].values()), 3
    )
    result["artifacts"].update(
        {
            "trace_preview": trace_metrics["overlay"],
            "assignment_preview": assignment["files"]["preview"],
            "surface_preview": reconstruction["files"]["preview"],
            "pixel_grid": reconstruction["files"]["grid"],
            "pixel_contours": reconstruction["files"]["final_contours"],
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Digitize one structural-map scan")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ocr-api-url", default="http://127.0.0.1:18080")
    parser.add_argument("--ocr-timeout", type=float, default=120.0)
    parser.add_argument("--interval", type=float)
    parser.add_argument("--trace-scale", type=float, default=0.6)
    args = parser.parse_args()
    result = run_pipeline(
        args.image,
        args.output_dir,
        ocr_api_url=args.ocr_api_url,
        interval=args.interval,
        trace_scale=args.trace_scale,
        ocr_timeout=args.ocr_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
