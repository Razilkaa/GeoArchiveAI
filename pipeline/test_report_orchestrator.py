from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report_orchestrator import ReportOrchestrator


class ReportOrchestratorTest(unittest.TestCase):
    def make_job(self, root: Path) -> Path:
        run_dir = root / "runs" / "demo"
        run_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "report_id": "demo",
            "source_root": str(root / "source"),
            "pages": [
                {"id": "page:00001", "role": "text"},
                {"id": "page:00002", "role": "graphic"},
            ],
            "queues": {},
            "summary": {"page_count": 2},
            "stages": {},
        }
        job = run_dir / "job.json"
        job.write_text(json.dumps(manifest), encoding="utf-8")
        bundle = {
            "report_id": "demo",
            "entities": {
                "structures": [
                    {"name": "Мейская структура", "evidence": ["page:00001"]},
                    {"name": "Мейнская структура", "evidence": ["page:00001"]},
                ],
                "wells": [],
                "maps": [],
                "key_results": [],
            },
        }
        (run_dir / "result_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        return job

    def test_map_unblocks_and_invalidates_qc_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            job = self.make_job(Path(temp))
            operator = ReportOrchestrator(job)
            first = operator.run()
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["agents"]["map"]["status"], "blocked")
            self.assertEqual(first["agents"]["qc"]["status"], "completed")
            self.assertEqual(first["agents"]["export"]["result"]["metrics"]["status"], "review")

            map_dir = job.parent / "map_agent"
            map_dir.mkdir()
            artifact = map_dir / "map.gpkg"
            artifact.write_bytes(b"geopackage-placeholder")
            (map_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "report_id": "demo",
                        "status": "completed",
                        "artifacts": [
                            {
                                "name": "vector_layers",
                                "path": str(artifact),
                                "media_type": "application/geopackage+sqlite3",
                            }
                        ],
                        "metrics": {"features": 4, "quality_status": "pass"},
                        "issues": [],
                    }
                ),
                encoding="utf-8",
            )

            second = ReportOrchestrator(job).run()
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["agents"]["map"]["status"], "completed")
            self.assertEqual(second["agents"]["qc"]["result"]["metrics"]["map_crosscheck"], "completed")
            self.assertEqual(second["agents"]["export"]["result"]["metrics"]["status"], "completed")
            qc_starts = [
                event
                for event in second["events"]
                if event["agent"] == "qc" and event["event"] == "started"
            ]
            self.assertEqual(len(qc_starts), 2)


if __name__ == "__main__":
    unittest.main()
