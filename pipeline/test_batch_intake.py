from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from batch_intake import _seven_zip, extract_archive, intake


class BatchIntakeTest(unittest.TestCase):
    def test_seven_zip_archive_is_extracted(self):
        try:
            executable = _seven_zip()
        except ValueError:
            self.skipTest("7-Zip is not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "payload"
            payload.mkdir()
            (payload / "page.jpg").write_bytes(b"scan")
            archive = root / "fund.7z"
            completed = subprocess.run(
                [str(executable), "a", str(archive), "payload"],
                cwd=root,
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.assertEqual(completed.returncode, 0)
            destination = extract_archive(archive, root / "out")
            self.assertEqual((destination / "payload" / "page.jpg").read_bytes(), b"scan")

    def test_upload_policy_is_applied_and_not_reported_as_unsupported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "inbox"
            staging = root / "staging"
            runs = root / "runs"
            inbox.mkdir()
            archive = inbox / "fund.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("report/page-1.jpg", b"scan")
            archive.with_suffix(".zip.policy.json").write_text(
                json.dumps({"external_models_allowed": True}), encoding="utf-8"
            )

            summary_path = intake(inbox, staging, runs)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["unsupported"], [])
            self.assertEqual(len(summary["registered"]), 1)
            privacy_path = Path(summary["registered"][0]["privacy"])
            privacy = json.loads(privacy_path.read_text(encoding="utf-8"))
            self.assertEqual(privacy["classification"], "accessible_archive")
            self.assertTrue(privacy["external_processing"])

    def test_accessible_archive_enables_processing_without_policy_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "inbox"
            report = inbox / "public-report"
            report.mkdir(parents=True)
            (report / "page-1.jpg").write_bytes(b"scan")

            summary_path = intake(inbox, root / "staging", root / "runs")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            privacy = json.loads(Path(summary["registered"][0]["privacy"]).read_text(encoding="utf-8"))

            self.assertEqual(privacy["classification"], "accessible_archive")
            self.assertTrue(privacy["external_processing"])


if __name__ == "__main__":
    unittest.main()
