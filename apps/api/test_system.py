from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from services.map_digitizer import PIPELINE_VERSION


class SystemApiTest(unittest.TestCase):
    @patch("app.routers.system.ocr_session.get")
    def test_services_exposes_map_pipeline_version(self, get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "ok"}
        get.return_value = response

        payload = TestClient(app).get("/api/services").json()

        self.assertEqual(
            payload["map_digitizer"]["pipeline_version"], PIPELINE_VERSION
        )
        self.assertEqual(
            payload["map_digitizer"]["acceptance_policy"],
            "dense_auto_accept_sparse_review",
        )


if __name__ == "__main__":
    unittest.main()
