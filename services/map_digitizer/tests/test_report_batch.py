from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.map_digitizer.report_batch import run_report_maps


class ReportBatchTest(unittest.TestCase):
    def test_processes_map_pages_skips_text_and_visual_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("L", (100, 100), 255).save(source / "map-a.png")
            Image.new("L", (100, 100), 255).save(source / "map-copy.png")
            Image.new("L", (100, 100), 128).save(source / "text.png")
            manifest = {
                "source_root": str(source),
                "pages": [
                    {"id": "page:1", "relative_path": "map-a.png", "content_type": "map"},
                    {"id": "page:2", "relative_path": "map-copy.png", "content_type": "map"},
                    {"id": "page:3", "relative_path": "text.png", "content_type": "text"},
                ],
            }
            manifest_path = root / "job.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            calls = []

            def fake_pipeline(image_path, output_dir, **_kwargs):
                calls.append(image_path)
                output_dir.mkdir(parents=True)
                preview = output_dir / "preview.png"
                preview.write_bytes(b"png")
                result = {
                    "status": "accepted",
                    "quality": {"status": "accepted"},
                    "artifacts": {"surface_preview": str(preview)},
                }
                (output_dir / "pipeline_result.json").write_text(json.dumps(result))
                return result

            result = run_report_maps(
                manifest_path, "http://ocr", pipeline_runner=fake_pipeline
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["quality_status"], "accepted")
        self.assertEqual(result["metrics"]["duplicates"], 1)
        self.assertEqual(result["metrics"]["candidates"], 2)

    def test_chart_candidate_cannot_be_auto_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chart.png"
            Image.new("L", (100, 100), 255).save(source)
            manifest = {
                "source_root": str(root),
                "pages": [
                    {"id": "page:1", "relative_path": source.name, "content_type": "chart"}
                ],
            }
            manifest_path = root / "job.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def fake_pipeline(_image_path, output_dir, **_kwargs):
                output_dir.mkdir(parents=True)
                return {"status": "accepted", "quality": {"status": "accepted"}, "artifacts": {}}

            result = run_report_maps(
                manifest_path, "http://ocr", pipeline_runner=fake_pipeline
            )

        self.assertEqual(result["jobs"][0]["status"], "review")
        self.assertIn(
            "routing_type_requires_review", result["jobs"][0]["quality"]["reasons"]
        )

    def test_preserves_external_georeferenced_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_agent = root / "map_agent"
            map_agent.mkdir()
            cps = root / "manual_surface.cps3"
            cps.write_text("grid")
            (map_agent / "result.json").write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {"name": "surface_grid", "path": str(cps)}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "job.json"
            manifest_path.write_text(
                json.dumps({"source_root": str(root), "pages": []}), encoding="utf-8"
            )

            result = run_report_maps(manifest_path, "http://ocr")

        self.assertEqual(result["metrics"]["preserved_artifacts"], 1)
        self.assertEqual(result["artifacts"][0]["name"], "surface_grid")


if __name__ == "__main__":
    unittest.main()
