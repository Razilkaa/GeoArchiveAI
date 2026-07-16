from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_result_bundle import build_bundle


class BuildResultBundleTest(unittest.TestCase):
    def test_map_result_without_summary_does_not_fail_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "job.json"
            manifest.write_text(
                json.dumps(
                    {
                        "report_id": "example",
                        "source_root": str(root),
                        "strategy": "full_text_fast_path",
                        "summary": {"page_count": 1},
                        "pages": [],
                        "stages": {"result_bundle": {"status": "pending", "metrics": {}}},
                    }
                ),
                encoding="utf-8",
            )
            ragflow = root / "ragflow.json"
            ragflow.write_text(json.dumps({"state": {"chunk_count": 2}}), encoding="utf-8")
            map_dir = root / "map_agent"
            map_dir.mkdir()
            map_result = map_dir / "result.json"
            map_result.write_text(json.dumps({"status": "pending"}), encoding="utf-8")

            output = build_bundle(manifest, ragflow, map_result_path=map_result)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["map_digitization"]["summary"], {})
        self.assertEqual(payload["summary"]["rag_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
