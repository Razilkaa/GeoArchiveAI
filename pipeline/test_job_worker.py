from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from job_worker import JobWorker, WorkerConfig, report_lock


class JobWorkerTest(unittest.TestCase):
    def _worker(self, root: Path) -> JobWorker:
        run = root / "runs" / "r1"
        run.mkdir(parents=True)
        manifest = {
            "report_id": "r1",
            "source_root": str(root / "source"),
            "pages": [],
            "queues": {"fast_ocr": [], "route_vlm": [], "deep_processing": []},
            "stages": {"page_routing": {"status": "completed"}},
        }
        (run / "job.json").write_text(json.dumps(manifest), encoding="utf-8")
        config = WorkerConfig(root / "creds", root / "token")
        return JobWorker(run / "job.json", config)

    def test_stage_checkpoint_skips_valid_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = self._worker(Path(temp))
            calls = []
            ok = worker._run_stage("markdown", lambda: True, lambda: calls.append("called"))
        self.assertTrue(ok)
        self.assertEqual(calls, [])
        self.assertTrue(worker.state["stages"]["markdown"]["cached"])

    def test_external_stage_is_allowed_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = self._worker(Path(temp))
            ok = worker._run_stage("agents", lambda: False, lambda: None, requires_external=True)
        self.assertTrue(ok)
        self.assertEqual(worker.state["stages"]["agents"]["status"], "completed")

    def test_report_lock_rejects_second_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            with report_lock(run):
                with self.assertRaisesRegex(RuntimeError, "already being processed"):
                    with report_lock(run):
                        pass


if __name__ == "__main__":
    unittest.main()
