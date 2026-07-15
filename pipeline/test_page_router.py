from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from page_router import apply_decisions, build_contact_sheet, load_preview, parse_decisions, route_pages
from report_factory import build_manifest, write_manifest


class PageRouterTest(unittest.TestCase):
    def test_contact_sheet_and_decisions_rebuild_queues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "report"
            root.mkdir()
            for index in range(3):
                Image.new("RGB", (800, 1000), "white").save(root / f"{index}.jpg")
            manifest = build_manifest(root)
            sheet = build_contact_sheet(manifest["pages"], root)
            decisions = [
                {"page_id": "page:00000", "kind": "toc", "confidence": 0.99},
                {"page_id": "page:00001", "kind": "text", "confidence": 0.95},
                {"page_id": "page:00002", "kind": "map", "confidence": 0.9},
            ]
            apply_decisions(manifest, decisions)

        self.assertEqual(sheet.size, (1680, 320))
        self.assertIn("page:00000", manifest["queues"]["fast_ocr"])
        self.assertIn("page:00002", manifest["queues"]["deep_processing"])
        self.assertEqual(manifest["pages"][2]["content_type"], "map")

    def test_parser_requires_every_page(self):
        batch = [{"id": "page:00000"}, {"id": "page:00001"}]
        text = '{"pages":[{"index":0,"kind":"text","confidence":0.8}]}'
        with self.assertRaisesRegex(ValueError, "omitted"):
            parse_decisions(text, batch)

    def test_preview_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "large.jpg"
            Image.new("RGB", (1600, 1200), "white").save(source)
            preview = load_preview(source, (200, 150))
        self.assertLessEqual(preview.width, 200)
        self.assertLessEqual(preview.height, 150)

    def test_explicit_path_role_wins_over_preview_mistake(self):
        manifest = {
            "source_root": ".",
            "pages": [
                {
                    "id": "page:00000",
                    "role": "toc",
                    "route_vlm": True,
                    "fast_ocr": False,
                    "fast_ocr_reasons": [],
                    "deep_processing": False,
                    "volume": None,
                    "page_number": 1,
                    "relative_path": "Оглавление/1.jpg",
                }
            ],
            "queues": {"fast_ocr": [], "route_vlm": [], "deep_processing": []},
            "summary": {},
        }
        apply_decisions(
            manifest,
            [{"page_id": "page:00000", "kind": "blank", "confidence": 0.9}],
        )
        self.assertEqual(manifest["pages"][0]["role"], "toc")
        self.assertIn("page:00000", manifest["queues"]["fast_ocr"])

        manifest["pages"][0].update(
            {
                "role": "graphic",
                "relative_path": "Графика/1.jpg",
                "route_vlm": True,
                "fast_ocr": False,
                "deep_processing": True,
            }
        )
        apply_decisions(
            manifest,
            [{"page_id": "page:00000", "kind": "toc", "confidence": 0.9}],
        )
        self.assertEqual(manifest["pages"][0]["role"], "graphic")
        self.assertNotIn("page:00000", manifest["queues"]["fast_ocr"])

    def test_parallel_batches_are_assembled_in_page_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "report"
            report.mkdir()
            for index in range(5):
                Image.new("RGB", (100, 120), "white").save(report / f"{index}.jpg")
            manifest_path = write_manifest(build_manifest(report), root / "runs")
            credentials = root / "credentials.txt"
            credentials.write_text("OPENAI_API_KEY=test", encoding="utf-8")
            calls = []

            def fake_classifier(batch, source_root, client, model):
                calls.append(batch[0]["id"])
                time.sleep(0.02 if batch[0]["id"] == "page:00000" else 0)
                return [
                    {"page_id": page["id"], "kind": "text", "confidence": 1.0}
                    for page in batch
                ]

            output = route_pages(
                manifest_path,
                credentials,
                batch_size=3,
                workers=2,
                classifier=fake_classifier,
            )
            payload = __import__("json").loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 2)
        self.assertEqual([item["page_id"] for item in payload["pages"]], [f"page:{i:05d}" for i in range(5)])


if __name__ == "__main__":
    unittest.main()
