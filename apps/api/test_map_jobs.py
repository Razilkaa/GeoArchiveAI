from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services import map_jobs


class MapJobsApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = SimpleNamespace(
            project_root=self.root,
            runs_root=self.root / "runs",
            ocr_api_url="http://ocr.test",
        )
        self.settings_patch = patch.object(map_jobs, "settings", self.settings)
        self.settings_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.settings_patch.stop()
        self.temp.cleanup()

    @patch.object(map_jobs.subprocess, "Popen")
    def test_upload_starts_background_digitization(self, popen):
        popen.return_value = Mock(pid=1234)

        response = self.client.post(
            "/api/maps/jobs?trace_scale=0.5",
            files={"file": ("sheet.jpg", io.BytesIO(b"jpeg"), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "running")
        directory = self.settings.runs_root / "map_jobs" / payload["job_id"]
        self.assertEqual((directory / "input.jpg").read_bytes(), b"jpeg")
        command = popen.call_args.args[0]
        self.assertIn("services.map_digitizer.pipeline", command)

    def test_upload_rejects_non_image(self):
        response = self.client.post(
            "/api/maps/jobs",
            files={"file": ("notes.txt", io.BytesIO(b"text"), "text/plain")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "unsupported_map_image")

    @patch.object(map_jobs, "active_map_job_count", return_value=2)
    def test_upload_respects_map_worker_capacity(self, _active_count):
        response = self.client.post(
            "/api/maps/jobs",
            files={"file": ("sheet.jpg", io.BytesIO(b"jpeg"), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"], "map_job_capacity_reached")

    def test_status_returns_completed_manifest(self):
        job_id = "a" * 32
        directory = self.settings.runs_root / "map_jobs" / job_id
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(
            json.dumps({"job_id": job_id, "status": "running"}), encoding="utf-8"
        )
        (directory / "pipeline_result.json").write_text(
            json.dumps({"status": "accepted", "quality": {"crossings": 0}}),
            encoding="utf-8",
        )

        response = self.client.get(f"/api/maps/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")
        self.assertEqual(response.json()["result"]["quality"]["crossings"], 0)

    def test_status_exposes_stable_artifact_urls(self):
        job_id = "f" * 32
        directory = self.settings.runs_root / "map_jobs" / job_id
        directory.mkdir(parents=True)
        preview = directory / "surface.png"
        preview.write_bytes(b"png")
        (directory / "pipeline_result.json").write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "artifacts": {"surface_preview": str(preview)},
                }
            ),
            encoding="utf-8",
        )

        payload = self.client.get(f"/api/maps/jobs/{job_id}").json()

        self.assertEqual(
            payload["artifact_urls"]["surface_preview"],
            f"/api/maps/jobs/{job_id}/artifacts/surface_preview",
        )

    def test_download_returns_pipeline_artifact(self):
        job_id = "d" * 32
        directory = self.settings.runs_root / "map_jobs" / job_id
        directory.mkdir(parents=True)
        preview = directory / "surface.png"
        preview.write_bytes(b"png-result")
        (directory / "pipeline_result.json").write_text(
            json.dumps({"artifacts": {"surface_preview": str(preview)}}),
            encoding="utf-8",
        )

        response = self.client.get(
            f"/api/maps/jobs/{job_id}/artifacts/surface_preview"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"png-result")
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_download_rejects_artifact_outside_job(self):
        job_id = "e" * 32
        directory = self.settings.runs_root / "map_jobs" / job_id
        directory.mkdir(parents=True)
        outside = self.root / "private.npz"
        outside.write_bytes(b"private")
        (directory / "pipeline_result.json").write_text(
            json.dumps({"artifacts": {"pixel_grid": str(outside)}}),
            encoding="utf-8",
        )

        response = self.client.get(f"/api/maps/jobs/{job_id}/artifacts/pixel_grid")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "map_artifact_not_found")

    @patch("app.routers.maps.run_georeference")
    def test_georeference_passes_validated_controls(self, georeference):
        georeference.return_value = {"status": "accepted", "target_crs": "EPSG:28421"}
        response = self.client.post(
            f"/api/maps/jobs/{'b' * 32}/georeference",
            json={
                "target_crs": "EPSG:28421",
                "control_points": [
                    {"pixel": [0, 0], "map": [1, 2]},
                    {"pixel": [100, 0], "map": [101, 2]},
                    {"pixel": [0, 100], "map": [1, 102]},
                ],
                "cell_size": 25,
                "name": "horizon_k",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_crs"], "EPSG:28421")
        self.assertEqual(georeference.call_args.kwargs["control_points"][1]["pixel"], (100.0, 0.0))

    def test_georeference_requires_three_controls(self):
        response = self.client.post(
            f"/api/maps/jobs/{'c' * 32}/georeference",
            json={
                "target_crs": "EPSG:28421",
                "control_points": [
                    {"pixel": [0, 0], "map": [1, 2]},
                    {"pixel": [100, 0], "map": [101, 2]},
                ],
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_georeference_cannot_approve_review_surface(self):
        combined = map_jobs.combine_georeference_quality(
            {"status": "accepted", "control_quality": {"p95_m": 0.1}},
            {
                "status": "review",
                "quality": {"reasons": ["sparse_reconstruction_requires_review"]},
            },
        )

        self.assertEqual(combined["status"], "review")
        self.assertEqual(combined["georeference_status"], "accepted")
        self.assertEqual(combined["source_surface_status"], "review")
        self.assertIn("source_surface_requires_review", combined["quality_reasons"])
        self.assertIn(
            "sparse_reconstruction_requires_review", combined["quality_reasons"]
        )

    def test_georeference_accepts_only_when_both_stages_accept(self):
        combined = map_jobs.combine_georeference_quality(
            {"status": "accepted"},
            {"status": "accepted", "quality": {"reasons": []}},
        )

        self.assertEqual(combined["status"], "accepted")
        self.assertEqual(combined["quality_reasons"], [])


if __name__ == "__main__":
    unittest.main()
