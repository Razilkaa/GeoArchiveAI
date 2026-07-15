from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
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


if __name__ == "__main__":
    unittest.main()
