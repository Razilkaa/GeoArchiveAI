from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from factory_runner import build_provenance_markdown, prepare_ocr_queue
from report_factory import build_manifest, write_manifest


class FactoryRunnerTest(unittest.TestCase):
    def test_queue_names_are_unique_and_markdown_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "report"
            for volume in ("Том 1", "Том 2"):
                directory = report / "Tекст" / volume
                directory.mkdir(parents=True)
                (directory / "00001.jpg").write_bytes(volume.encode("utf-8"))
            manifest = build_manifest(report)
            manifest_path = write_manifest(manifest, root / "runs")
            queue_path = prepare_ocr_queue(manifest_path)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            self.assertEqual(len({item["input_name"] for item in queue}), 2)

            output_dir = queue_path.parent / "output"
            output_dir.mkdir()
            for item in queue:
                result = {"rec_texts": [f"text from {item['page_id']}"], "rec_scores": [0.9]}
                (output_dir / f"{Path(item['input_name']).stem}_res.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
            markdown = build_provenance_markdown(manifest_path).read_text(encoding="utf-8")

        self.assertEqual(markdown.count("[SOURCE_PAGE:"), 2)
        self.assertIn("Том 1", markdown)
        self.assertIn("Том 2", markdown)

    def test_selected_pdf_page_is_materialized_for_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "report"
            text = report / "Текст"
            text.mkdir(parents=True)
            pdf = text / "Том 1.pdf"
            document = fitz.open()
            document.new_page().insert_text((72, 72), "archive report")
            document.save(pdf)
            document.close()
            manifest = build_manifest(report)
            manifest_path = write_manifest(manifest, root / "runs")
            queue_path = prepare_ocr_queue(manifest_path)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            rendered = queue_path.parent / "input" / queue[0]["input_name"]
            rendered_suffix = rendered.suffix
            rendered_size = rendered.stat().st_size

        self.assertEqual(queue[0]["pdf_page"], 0)
        self.assertEqual(rendered_suffix, ".jpg")
        self.assertTrue(rendered_size > 0)

    def test_markdown_splits_large_page_without_losing_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "report" / "Текст" / "Том 1"
            report.mkdir(parents=True)
            (report / "00001.jpg").write_bytes(b"scan")
            manifest = build_manifest(root / "report")
            manifest_path = write_manifest(manifest, root / "runs")
            queue = json.loads(prepare_ocr_queue(manifest_path).read_text(encoding="utf-8"))
            output_dir = manifest_path.parent / "fast_ocr" / "output"
            output_dir.mkdir()
            lines = [f"line {index} " + "x" * 500 for index in range(30)]
            result_path = output_dir / f"{Path(queue[0]['input_name']).stem}_res.json"
            result_path.write_text(json.dumps({"rec_texts": lines}), encoding="utf-8")

            markdown = build_provenance_markdown(manifest_path).read_text(encoding="utf-8")

        self.assertGreater(markdown.count("## page:"), 1)
        self.assertEqual(markdown.count("[SOURCE_PAGE:"), len(lines))
        blocks = [block for block in markdown.split("## page:") if block]
        self.assertLess(max(map(len, blocks)), 8000)


if __name__ == "__main__":
    unittest.main()
