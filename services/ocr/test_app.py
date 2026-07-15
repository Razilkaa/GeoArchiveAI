from __future__ import annotations

import io
import unittest

from PIL import Image

from services.ocr.app import _write_normalized_image


class ImageNormalizationTest(unittest.TestCase):
    def test_tiff_is_normalized_to_rgb_jpeg(self) -> None:
        source = io.BytesIO()
        Image.new("L", (24, 16), color=180).save(source, format="TIFF")

        normalized = io.BytesIO()
        _write_normalized_image(source.getvalue(), normalized)

        normalized.seek(0)
        with Image.open(normalized) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (24, 16))


if __name__ == "__main__":
    unittest.main()
