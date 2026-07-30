from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from services.map_digitizer import PIPELINE_VERSION
from services.map_digitizer.report_batch import reusable_result, run_report_maps


def test_unverified_similarity_georeference_is_not_reused(tmp_path: Path) -> None:
    source = tmp_path / "map.png"
    Image.new("L", (16, 16), "white").save(source)
    stat = source.stat()
    result = tmp_path / "pipeline_result.json"
    result.write_text(
        json.dumps(
            {
                "version": PIPELINE_VERSION,
                "source": str(source),
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "status": "review",
                "stages": {
                    "georeference": {"method": "survey_profile_similarity"}
                },
            }
        ),
        encoding="utf-8",
    )

    assert reusable_result(result, source) is None


def test_report_map_pages_run_in_parallel(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_root.mkdir()
    run_dir.mkdir()
    pages = []
    random = np.random.default_rng(452289)
    for index in range(4):
        image_path = source_root / f"page-{index}.png"
        pixels = random.integers(0, 256, size=(64, 64), dtype=np.uint8)
        Image.fromarray(pixels).save(image_path)
        pages.append(
            {
                "id": f"page:{index:05d}",
                "relative_path": image_path.name,
                "content_type": "map",
            }
        )

    manifest_path = run_dir / "job.json"
    manifest_path.write_text(
        json.dumps(
            {
                "report_id": "452289",
                "source_root": str(source_root),
                "pages": pages,
            }
        ),
        encoding="utf-8",
    )

    lock = threading.Lock()
    active = 0
    peak_active = 0

    def runner(source: Path, output_dir: Path, **_: object) -> dict:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"status": "not_applicable", "quality": {}, "artifacts": {}}

    result = run_report_maps(
        manifest_path,
        "http://ocr.invalid",
        pipeline_runner=runner,
        workers=4,
    )

    assert result["status"] == "completed"
    assert result["metrics"]["processed"] == 4
    assert result["metrics"]["workers"] == 4
    assert peak_active >= 2
