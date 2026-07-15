from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.automation import InboxMonitor, start_pending_reports
from app.services.jobs import start_report


class AutomationTest(unittest.TestCase):
    @patch("app.services.automation.start_report")
    @patch("app.services.automation.list_reports")
    def test_queue_respects_parallel_report_limit(self, list_reports, start):
        list_reports.return_value = [
            {"report_id": "running", "worker_status": "running"},
            {"report_id": "first", "worker_status": "pending"},
            {"report_id": "second", "worker_status": "pending"},
        ]
        start.return_value = {"report_id": "first", "status": "accepted"}

        result = start_pending_reports(max_reports=2)

        start.assert_called_once_with("first", [])
        self.assertEqual(result["started"], ["first"])
        self.assertEqual(result["queued"], ["second"])
        self.assertEqual(result["active"], 2)

    @patch("app.services.jobs.report_run_dir")
    def test_start_is_idempotent_while_worker_lock_exists(self, run_dir):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            lock = directory / "worker" / "worker.lock"
            lock.parent.mkdir()
            lock.write_text("{}", encoding="utf-8")
            run_dir.return_value = directory

            result = start_report("report-1", [])

        self.assertEqual(result["status"], "running")
        self.assertIsNone(result["pid"])

    @patch("app.services.automation.reconcile_now")
    def test_monitor_waits_for_stable_inbox_before_registration(self, reconcile):
        reconcile.return_value = {"status": "completed"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "report.tif").write_bytes(b"scan")
            fake_settings = SimpleNamespace(inbox_root=root, auto_intake_settle_s=1)
            with patch("app.services.automation.settings", fake_settings):
                monitor = InboxMonitor()
                monitor.tick()
                monitor.observed_since -= 2
                monitor.tick()

        self.assertEqual(reconcile.call_args_list[0].kwargs, {"register": False})
        self.assertEqual(reconcile.call_args_list[1].kwargs, {"register": True})


if __name__ == "__main__":
    unittest.main()
