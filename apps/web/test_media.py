from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image

from apps.web.media import image_preview


class ImagePreviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_raster_preview_is_bounded_rgb(self):
        path = self.root / "map.png"
        Image.new("RGBA", (1000, 500), (20, 80, 60, 255)).save(path)

        preview = image_preview(path, max_size=(200, 200))

        self.assertIsNotNone(preview)
        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(preview.size, (200, 100))

    def test_pdf_preview_renders_first_page(self):
        path = self.root / "map.pdf"
        with fitz.open() as document:
            page = document.new_page(width=600, height=400)
            page.insert_text((72, 72), "Structural map")
            document.save(path)

        preview = image_preview(path, max_size=(300, 300))

        self.assertIsNotNone(preview)
        self.assertEqual(preview.mode, "RGB")
        self.assertLessEqual(preview.width, 300)
        self.assertLessEqual(preview.height, 300)

    def test_invalid_asset_returns_none(self):
        path = self.root / "broken.pdf"
        path.write_text("not a pdf", encoding="utf-8")

        self.assertIsNone(image_preview(path))
        self.assertIsNone(image_preview(self.root / "missing.jpg"))


if __name__ == "__main__":
    unittest.main()
