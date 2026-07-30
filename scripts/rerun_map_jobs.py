"""Rerun the map digitization pipeline over existing report jobs.

Runs sequentially with capped BLAS threads: two full-resolution sheets do not
fit in memory side by side on the workstation.  OCR is reused from each job's
cached ``ocr.json``.

Usage:
    python scripts/rerun_map_jobs.py runs/384092/map_agent/jobs [--pages page_00186 ...]
        [--fallback-interval 0.2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument("--pages", nargs="*")
    parser.add_argument("--fallback-interval", type=float)
    parser.add_argument("--survey-shape", type=Path)
    parser.add_argument("--inventory-id")
    args = parser.parse_args()

    from services.map_digitizer.pipeline import run_pipeline

    from services.map_digitizer.survey_profiles import find_survey_shape

    shapes_dir = Path(__file__).resolve().parents[1] / "shapes"

    jobs = sorted(
        job
        for job in args.jobs_dir.iterdir()
        if job.is_dir() and (job / "pipeline_result.json").exists()
    )
    if args.pages:
        wanted = set(args.pages)
        jobs = [job for job in jobs if job.name in wanted]

    summary = []
    for job in jobs:
        previous = json.loads(
            (job / "pipeline_result.json").read_text(encoding="utf-8")
        )
        inventory_id = args.inventory_id
        if not inventory_id:
            inventory_id = next(
                (part for part in Path(previous["source"]).parts if part.isdigit()),
                None,
            )
        # Each region ships its own survey inventory, so resolve the shapefile
        # from the report number rather than assuming one file fits every run.
        survey_shape = args.survey_shape
        if survey_shape is None and inventory_id and shapes_dir.exists():
            survey_shape = find_survey_shape(inventory_id, shapes_dir)
        try:
            result = run_pipeline(
                Path(previous["source"]),
                job,
                fallback_interval=args.fallback_interval,
                survey_shape_path=survey_shape,
                inventory_id=inventory_id,
            )
            reconstruction = (
                result.get("stages", {})
                .get("reconstruction", {})
                .get("metrics", {})
            )
            item = {
                "page": job.name,
                "status": result["status"],
                "reasons": result.get("quality", {}).get("reasons"),
                "value_sources": reconstruction.get("value_source_distribution"),
                "propagation": reconstruction.get("profile_propagation"),
                "rmse_m": reconstruction.get("grid_quality", {}).get(
                    "constraint_rmse_m"
                ),
                "crossing_pairs": reconstruction.get("topology", {}).get(
                    "crossing_pairs"
                ),
                "georeference": (
                    {
                        "residual_median_m": stage["metrics"]["residual_median_m"],
                        "aligned": stage["metrics"]["icp"]["aligned_numbers"],
                    }
                    if (stage := result["stages"].get("georeference", {})).get(
                        "status"
                    )
                    == "complete"
                    else stage.get("status")
                ),
            }
        except Exception as error:
            item = {
                "page": job.name,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            traceback.print_exc()
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    out = args.jobs_dir / "rerun_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
