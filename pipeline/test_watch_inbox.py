from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from watch_inbox import fingerprint, inbox_is_ready


class WatchInboxTest(unittest.TestCase):
    def test_nested_copy_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            inbox = Path(temp)
            report = inbox / "report"
            volume = report / "volume"
            volume.mkdir(parents=True)
            (volume / "001.jpg").write_bytes(b"first")
            before = fingerprint(inbox)

            (volume / "002.jpg").write_bytes(b"second-page")
            after = fingerprint(inbox)

            self.assertNotEqual(before, after)
            self.assertEqual(after[0][1], 2)
            self.assertEqual(after[0][2], len(b"first") + len(b"second-page"))

    def test_ready_walks_nested_files(self):
        with tempfile.TemporaryDirectory() as temp:
            inbox = Path(temp)
            nested = inbox / "report" / "volume"
            nested.mkdir(parents=True)
            (nested / "001.jpg").write_bytes(b"page")
            self.assertTrue(inbox_is_ready(inbox))


if __name__ == "__main__":
    unittest.main()
