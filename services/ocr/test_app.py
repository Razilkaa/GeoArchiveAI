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

    def test_faded_scan_contrast_is_expanded(self) -> None:
        source_image = Image.new("L", (40, 20), color=220)
        for x in range(20):
            for y in range(20):
                source_image.putpixel((x, y), 190)
        source = io.BytesIO()
        source_image.save(source, format="PNG")

        normalized = io.BytesIO()
        _write_normalized_image(source.getvalue(), normalized, enhanced=True)

        normalized.seek(0)
        with Image.open(normalized) as image:
            minimum, maximum = image.convert("L").getextrema()
            self.assertLess(minimum, 25)
            self.assertGreater(maximum, 230)


if __name__ == "__main__":
    unittest.main()
