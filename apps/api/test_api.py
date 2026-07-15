from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.routers import reports as report_router
from app.services import reports as report_service


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runs = Path(self.temp.name)
        for report_id in ("375392", "CD отчет"):
            directory = self.runs / report_id
            directory.mkdir()
            (directory / "job.json").write_text(
                json.dumps(
                    {
                        "report_id": report_id,
                        "source_root": str(directory),
                        "summary": {"page_count": 3},
                    }
                ),
                encoding="utf-8",
            )
        self.patch = patch.object(report_service, "settings", SimpleNamespace(runs_root=self.runs))
        self.patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_reports_include_unicode_identifier(self):
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 2)
        self.assertIn("CD отчет", {item["report_id"] for item in response.json()["reports"]})

    def test_report_status_rejects_path_traversal(self):
        response = self.client.get("/api/reports/%2E%2E/status")
        self.assertIn(response.status_code, {400, 404})

    def test_openapi_exposes_intake_and_run_contracts(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/reports/upload", paths)
        self.assertIn("/api/reports/{report_id}/run", paths)
        self.assertIn("/api/queue", paths)
        self.assertIn("/api/search", paths)

    @patch.object(report_router, "corpus_answer_service")
    def test_corpus_search_endpoint(self, service_factory):
        service_factory.return_value.ask.return_value = {
            "question": "Где были притоки?",
            "answer": "В отчёте 375392 [источник 1].",
            "evidence": [{"report_id": "375392"}],
        }

        response = self.client.post(
            "/api/search", json={"question": "Где были притоки?", "mode": "live"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evidence"][0]["report_id"], "375392")

    def test_worker_lock_is_reported_as_running(self):
        lock = self.runs / "375392" / "worker" / "worker.lock"
        lock.parent.mkdir()
        lock.write_text("{}", encoding="utf-8")

        reports = self.client.get("/api/reports").json()["reports"]

        report = next(item for item in reports if item["report_id"] == "375392")
        self.assertEqual(report["worker_status"], "running")

    def test_stale_running_state_with_completed_core_is_normalized(self):
        directory = self.runs / "375392"
        worker = directory / "worker"
        worker.mkdir()
        stages = {
            name: {"status": "completed"}
            for name in ("page_routing", "fast_ocr", "markdown", "ragflow", "bundle")
        }
        (worker / "state.json").write_text(
            json.dumps({"status": "running", "stages": stages}), encoding="utf-8"
        )

        response = self.client.get("/api/reports/375392/status")

        self.assertEqual(response.json()["worker"]["status"], "completed")

    def test_maps_endpoint_returns_only_detected_maps(self):
        directory = self.runs / "375392"
        source = directory / "map.jpg"
        source.write_bytes(b"image")
        manifest = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        manifest["pages"] = [
            {"id": "page:00001", "relative_path": "map.jpg", "content_type": "map"},
            {"id": "page:00002", "relative_path": "text.jpg", "content_type": "text"},
        ]
        (directory / "job.json").write_text(json.dumps(manifest), encoding="utf-8")

        response = self.client.get("/api/reports/375392/maps")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["sources"]), 1)
        self.assertEqual(response.json()["sources"][0]["page_id"], "page:00001")
        self.assertEqual(response.json()["sources"][0]["media_type"], "image/jpeg")
        self.assertEqual(response.json()["status"], "not_digitized")

    def test_maps_endpoint_reports_pdf_media_type(self):
        directory = self.runs / "375392"
        source = directory / "map.pdf"
        source.write_bytes(b"%PDF-1.4")
        manifest = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        manifest["pages"] = [
            {"id": "page:00001", "relative_path": "map.pdf", "content_type": "map"},
        ]
        (directory / "job.json").write_text(json.dumps(manifest), encoding="utf-8")

        response = self.client.get("/api/reports/375392/maps")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["media_type"], "application/pdf")

    @patch("app.routers.reports.reconcile_now")
    def test_scan_registers_and_starts_reports(self, reconcile):
        reconcile.return_value = {
            "registered": [{"report_id": "new-report"}],
            "unsupported": [],
            "started": ["new-report"],
            "queued": ["next-report"],
        }

        response = self.client.post("/api/intake/scan")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["started"], 1)
        self.assertEqual(response.json()["queued"], 1)


if __name__ == "__main__":
    unittest.main()
