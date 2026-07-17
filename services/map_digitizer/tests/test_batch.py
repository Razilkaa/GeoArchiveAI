from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.map_digitizer.batch import case_directory, discover_images, reusable_result
from services.map_digitizer.pipeline import file_sha256


class BatchTest(unittest.TestCase):
    def test_discovers_supported_images_and_uses_stable_case_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "map 01.JPG"
            image.touch()
            (root / "notes.txt").touch()
            discovered = discover_images([root])
            first = case_directory(root / "out", image.resolve())
            second = case_directory(root / "out", image.resolve())
        self.assertEqual(discovered, [image.resolve()])
        self.assertEqual(first, second)

    def test_reuses_only_terminal_result_for_same_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "map.jpg"
            source.touch()
            result = root / "pipeline_result.json"
            result.write_text(
                json.dumps({"source": str(source.resolve()), "status": "accepted"}),
                encoding="utf-8",
            )
            self.assertIsNotNone(reusable_result(result, source))
            result.write_text(
                json.dumps({"source": str(source.resolve()), "status": "failed"}),
                encoding="utf-8",
            )
            self.assertIsNone(reusable_result(result, source))

    def test_changed_source_invalidates_hashed_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "map.jpg"
            source.write_bytes(b"first")
            result = root / "pipeline_result.json"
            result.write_text(
                json.dumps(
                    {
                        "source": str(source.resolve()),
                        "source_sha256": file_sha256(source),
                        "status": "accepted",
                    }
                ),
                encoding="utf-8",
            )
            source.write_bytes(b"second")

            self.assertIsNone(reusable_result(result, source))


if __name__ == "__main__":
    unittest.main()
