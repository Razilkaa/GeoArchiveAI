from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from report_factory import build_manifest, ensure_full_text_strategy


class ReportFactoryTest(unittest.TestCase):
    def test_plan_ocr_includes_every_text_page_and_toc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "report-1"
            text = root / "Tекст" / "Том 1"
            graphics = root / "Графика" / "Том 2"
            toc = graphics / "Оглавление"
            sample = root / "Графика" / "Том 2, 3 (выборка)"
            for directory in (text, graphics, toc, sample):
                directory.mkdir(parents=True, exist_ok=True)
            for page in range(40):
                (text / f"{page:05d}.jpg").write_bytes(b"scan")
            (graphics / "21.jpg").write_bytes(b"map")
            (toc / "1.jpg").write_bytes(b"toc")
            (sample / "21.bmp").write_bytes(b"duplicate")

            manifest = build_manifest(root)

        self.assertEqual(manifest["summary"]["fast_ocr_pages"], 41)
        self.assertEqual(manifest["summary"]["deep_processing_pages"], 1)
        sample_page = next(page for page in manifest["pages"] if page["role"] == "sample")
        self.assertFalse(sample_page["route_vlm"])
        self.assertFalse(sample_page["deep_processing"])
        toc_page = next(page for page in manifest["pages"] if page["role"] == "toc")
        self.assertIn("explicit_toc", toc_page["fast_ocr_reasons"])
        text_pages = [page for page in manifest["pages"] if page["role"] == "text"]
        self.assertTrue(all(page["fast_ocr"] for page in text_pages))
        self.assertTrue(all("full_text_corpus" in page["fast_ocr_reasons"] for page in text_pages))

    def test_pdf_pages_are_inventoried_without_full_render(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "report"
            root.mkdir()
            pdf = root / "Том 1.pdf"
            document = fitz.open()
            document.new_page().insert_text((72, 72), "page one")
            document.new_page().insert_text((72, 72), "page two")
            document.save(pdf)
            document.close()
            manifest = build_manifest(root)

        self.assertEqual(manifest["summary"]["page_count"], 2)
        self.assertEqual([page["pdf_page"] for page in manifest["pages"]], [0, 1])
        self.assertTrue(all(page["source_kind"] == "pdf" for page in manifest["pages"]))

    def test_old_selective_manifest_is_upgraded_idempotently(self):
        manifest = {
            "strategy": "selective_fast_path",
            "pages": [
                {"id": "page:00001", "role": "text", "fast_ocr": False, "fast_ocr_reasons": []},
                {"id": "page:00002", "role": "graphic", "fast_ocr": False, "fast_ocr_reasons": []},
            ],
            "queues": {"fast_ocr": []},
            "summary": {"fast_ocr_pages": 0},
        }
        self.assertTrue(ensure_full_text_strategy(manifest))
        self.assertFalse(ensure_full_text_strategy(manifest))
        self.assertEqual(manifest["queues"]["fast_ocr"], ["page:00001"])
        self.assertEqual(manifest["pages"][0]["fast_ocr_reasons"], ["full_text_corpus"])


if __name__ == "__main__":
    unittest.main()
