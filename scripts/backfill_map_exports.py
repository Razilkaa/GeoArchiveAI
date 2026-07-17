"""Backfill clean previews and local CPS-3 files without rerunning OCR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.map_digitizer.local_exports import materialize_local_exports
from services.map_digitizer.report_batch import run_report_maps


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def backfill(report_id: str, ocr_api_url: str) -> dict[str, int]:
    run_dir = PROJECT_ROOT / "runs" / report_id
    result_path = run_dir / "map_agent" / "result.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    updated = 0
    for job in report.get("jobs", []):
        if job.get("status") not in {"accepted", "review"}:
            continue
        pipeline_path = resolve(job["result"])
        payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
        artifacts = payload.setdefault("artifacts", {})
        grid_path = artifacts.get("pixel_grid")
        contours_path = artifacts.get("pixel_contours")
        if not grid_path or not contours_path:
            continue
        exports = materialize_local_exports(
            resolve(grid_path),
            resolve(contours_path),
            pipeline_path.parent / "surface",
        )
        artifacts.update(exports)
        temporary = pipeline_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(pipeline_path)
        updated += 1

    run_report_maps(run_dir / "job.json", ocr_api_url)
    return {"report": report_id, "updated": updated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--ocr-api-url", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    for report_id in args.reports:
        print(json.dumps(backfill(report_id, args.ocr_api_url), ensure_ascii=False))


if __name__ == "__main__":
    main()
