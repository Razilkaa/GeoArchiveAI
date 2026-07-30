"""Publish per-sheet digitization artifacts from job dirs into results/<report>/sheet_NN.

Restores the review layout that earlier sessions copied by hand: each sheet
folder holds the clean digitized map, previews, grids and vector layers under
stable names, so /results stays comparable between pipeline runs.

Usage:
    python scripts/publish_map_results.py runs/384092/map_agent/jobs results/384092
        [--pages page_00184 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

PUBLISH_MAP = {
    "surface_clean_preview": "digitized_map.png",
    "surface_preview": "surface_preview.png",
    "pixel_contours": "digitized_source_contours_pixels.geojson",
    "interpolated_contours": "reconstructed_contours_pixels.geojson",
    "local_cps3": "surface_local_pixels.cps3",
}
EXTRA_SURFACE_FILES = {
    "digitized_isolines_by_level.geojson": "isolines_by_level_pixels.geojson",
    "digitized_isolines_overlay.png": "digitized_isolines_overlay.png",
    "profile_points.csv": "profile_points_pixels.csv",
    "trace_guided_surface_metrics.json": "surface_metrics.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--pages", nargs="*")
    args = parser.parse_args()

    published = []
    for job in sorted(args.jobs_dir.iterdir()):
        result_path = job / "pipeline_result.json"
        if not job.is_dir() or not result_path.exists():
            continue
        if args.pages and job.name not in set(args.pages):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        source = Path(result.get("source", ""))
        match = re.search(r"(\d+)", source.stem)
        if not match:
            continue
        sheet_dir = args.results_dir / f"sheet_{int(match.group(1))}"
        artifacts = result.get("artifacts", {})
        if "surface_clean_preview" not in artifacts:
            continue
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for name, target in PUBLISH_MAP.items():
            path = Path(str(artifacts.get(name) or ""))
            if path.exists():
                shutil.copy2(path, sheet_dir / target)
        surface_dir = job / "surface"
        for original, target in EXTRA_SURFACE_FILES.items():
            path = surface_dir / original
            if path.exists():
                shutil.copy2(path, sheet_dir / target)
        shutil.copy2(result_path, sheet_dir / "pipeline_result.json")
        published.append(f"{job.name} -> {sheet_dir}")
    print("\n".join(published) if published else "nothing published")


if __name__ == "__main__":
    main()
