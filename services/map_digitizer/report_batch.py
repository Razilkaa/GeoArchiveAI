"""Run map digitization for map pages already classified in a report manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import fitz
import numpy as np
from PIL import Image

from services.map_digitizer.batch import reusable_result
from services.map_digitizer.pipeline import run_pipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CANDIDATE_CONTENT_TYPES = {"map", "chart"}


def map_pages(manifest: dict) -> list[dict]:
    return [
        page
        for page in manifest.get("pages", [])
        if page.get("content_type") in CANDIDATE_CONTENT_TYPES
    ]


def report_map_signature(manifest: dict) -> str:
    records = [
        {
            "id": page.get("id"),
            "relative_path": page.get("relative_path"),
            "pdf_page": page.get("pdf_page"),
        }
        for page in map_pages(manifest)
    ]
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _safe_page_id(page_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in page_id)


def prepare_page_image(page: dict, source_root: Path, cache_dir: Path) -> Path:
    source = source_root / str(page.get("relative_path") or "")
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.casefold() in IMAGE_SUFFIXES:
        return source
    if source.suffix.casefold() != ".pdf":
        raise ValueError(f"Unsupported map source: {source.suffix}")
    page_index = int(page.get("pdf_page") or 0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{_safe_page_id(str(page.get('id')))}.png"
    if output.exists() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return output
    with fitz.open(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise IndexError(f"PDF page {page_index} is out of range")
        pixmap = document[page_index].get_pixmap(matrix=fitz.Matrix(4.0, 4.0), alpha=False)
        pixmap.save(output)
    return output


def perceptual_hash(image_path: Path, size: int = 16) -> np.ndarray:
    with Image.open(image_path) as image:
        sample = np.asarray(
            image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return (sample[:, 1:] > sample[:, :-1]).ravel()


def is_visual_duplicate(fingerprint: np.ndarray, fingerprints: list[np.ndarray]) -> bool:
    return any(float(np.mean(fingerprint != existing)) <= 0.04 for existing in fingerprints)


def run_report_maps(
    manifest_path: Path,
    ocr_api_url: str,
    *,
    pipeline_runner: Callable[..., dict] = run_pipeline,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    source_root = Path(str(manifest.get("source_root") or ""))
    output_root = run_dir / "map_agent"
    jobs_root = output_root / "jobs"
    cache_root = output_root / "page_cache"
    existing_result_path = output_root / "result.json"
    try:
        existing_result = json.loads(existing_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing_result = {}
    output_root.mkdir(parents=True, exist_ok=True)
    jobs_root_resolved = jobs_root.resolve()
    preserved_artifacts = []
    for item in existing_result.get("artifacts", []):
        path = Path(str(item.get("path") or ""))
        if not path.exists():
            continue
        try:
            managed = path.resolve().is_relative_to(jobs_root_resolved)
        except (OSError, ValueError):
            managed = False
        if not managed:
            preserved_artifacts.append(item)
    fingerprints: list[np.ndarray] = []
    jobs = []
    artifacts = []
    issues = []

    def checkpoint(status: str) -> dict:
        statuses = [job["status"] for job in jobs if job["status"] != "duplicate"]
        quality_status = (
            "review"
            if "review" in statuses or "failed" in statuses
            else "accepted"
            if "accepted" in statuses
            else "not_applicable"
        )
        generated_paths = {str(item.get("path")) for item in artifacts}
        combined_artifacts = [
            *artifacts,
            *[
                item
                for item in preserved_artifacts
                if str(item.get("path")) not in generated_paths
            ],
        ]
        payload = {
            "producer": "services.map_digitizer.report_batch",
            "status": status,
            "quality_status": quality_status,
            "manifest_signature": report_map_signature(manifest),
            "metrics": {
                "candidates": len(map_pages(manifest)),
                "processed": len(statuses),
                "duplicates": sum(job["status"] == "duplicate" for job in jobs),
                "accepted": statuses.count("accepted"),
                "review": statuses.count("review"),
                "not_applicable": statuses.count("not_applicable"),
                "failed": statuses.count("failed"),
                "reused": sum(bool(job.get("reused")) for job in jobs),
                "preserved_artifacts": len(preserved_artifacts),
            },
            "jobs": jobs,
            "artifacts": combined_artifacts,
            "issues": issues,
        }
        temporary = output_root / "result.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_root / "result.json")
        return payload

    checkpoint("running")
    for page in map_pages(manifest):
        page_id = str(page.get("id") or "unknown")
        try:
            source = prepare_page_image(page, source_root, cache_root)
            fingerprint = perceptual_hash(source)
            if is_visual_duplicate(fingerprint, fingerprints):
                jobs.append({"page_id": page_id, "status": "duplicate", "source": str(source)})
                checkpoint("running")
                continue
            fingerprints.append(fingerprint)
            job_dir = jobs_root / _safe_page_id(page_id)
            result = reusable_result(job_dir / "pipeline_result.json", source)
            reused = result is not None
            if result is None:
                result = pipeline_runner(
                    source,
                    job_dir,
                    ocr_api_url=ocr_api_url,
                )
            routed_type = str(page.get("content_type") or "map")
            effective_status = str(result.get("status"))
            effective_quality = dict(result.get("quality") or {})
            if routed_type != "map" and effective_status == "accepted":
                effective_status = "review"
                effective_quality["status"] = "review"
                effective_quality["reasons"] = [
                    *effective_quality.get("reasons", []),
                    "routing_type_requires_review",
                ]
            jobs.append(
                {
                    "page_id": page_id,
                    "routing_type": routed_type,
                    "status": effective_status,
                    "source": str(source),
                    "result": str(job_dir / "pipeline_result.json"),
                    "quality": effective_quality,
                    "reused": reused,
                }
            )
            artifacts.append(
                {
                    "name": "source_map",
                    "label": f"Исходная карта · {source.name}",
                    "path": str(source),
                    "media_type": "image/png" if source.suffix.casefold() == ".png" else "image/jpeg",
                    "page_id": page_id,
                }
            )
            if effective_status in {"accepted", "review"}:
                for name, label, media_type in (
                    ("surface_preview", "Оцифрованная поверхность", "image/png"),
                    ("pixel_grid", "Grid в пиксельной системе", "application/octet-stream"),
                    ("pixel_contours", "Оцифрованные изолинии", "application/geo+json"),
                ):
                    path = result.get("artifacts", {}).get(name)
                    if path:
                        artifacts.append(
                            {
                                "name": name,
                                "label": f"{label} · {source.name}",
                                "path": path,
                                "media_type": media_type,
                                "page_id": page_id,
                            }
                        )
        except Exception as error:
            jobs.append({"page_id": page_id, "status": "failed", "error": type(error).__name__})
            issues.append(
                {
                    "code": "map_page_failed",
                    "message": f"{page_id}: {type(error).__name__}: {str(error)[:300]}",
                    "severity": "warning",
                }
            )
        checkpoint("running")

    return checkpoint("completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Digitize classified map pages in one report")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ocr-api-url", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    print(
        json.dumps(
            run_report_maps(args.manifest, args.ocr_api_url),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
