from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.map_digitizer.ocr_client import deduplicate_lines, request_ocr, tile_starts


class OcrClientTest(unittest.TestCase):
    def test_tile_starts_cover_image_without_tiny_edge_tiles(self):
        self.assertEqual(tile_starts(4000, 5000), [0])
        self.assertEqual(tile_starts(8000, 5000), [0, 3000])
        self.assertEqual(tile_starts(12_000, 5000), [0, 3500, 7000])

    def test_deduplicates_overlap_and_keeps_higher_confidence(self):
        lines = [
            {"text": "-2.8", "score": 0.8, "polygon": [[10, 10], [30, 10], [30, 20], [10, 20]]},
            {"text": "-2.8", "score": 0.99, "polygon": [[11, 10], [31, 10], [31, 20], [11, 20]]},
            {"text": "-2.9", "score": 0.9, "polygon": [[100, 10], [120, 10], [120, 20], [100, 20]]},
        ]
        result = deduplicate_lines(lines)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["score"], 0.99)

    def test_large_image_is_downscaled_once_and_coordinates_are_restored(self):
        calls = []

        def fake_post(path: Path, _url: str, _timeout: float) -> dict:
            calls.append(path.name)
            return {
                "engine": "test",
                "lines": [
                    {
                        "text": path.stem,
                        "score": 0.9,
                        "polygon": [[1, 2], [10, 2], [10, 8], [1, 8]],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "map.png"
            Image.new("L", (100, 80), 255).save(image_path)
            result = request_ocr(
                image_path,
                "http://ocr",
                post=fake_post,
                max_direct_pixels=1,
                tile_size=50,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["preprocessing"], "downscaled")
        self.assertEqual(result["lines"][0]["polygon"][0], [2.0, 4.0])

    def test_failed_downscale_falls_back_to_overlapping_tiles(self):
        import requests

        calls = []

        def fake_post(path: Path, _url: str, _timeout: float) -> dict:
            calls.append(path.name)
            if path.name == "normalized.png":
                raise requests.HTTPError("oom")
            return {
                "engine": "test",
                "lines": [
                    {"text": path.stem, "score": 0.9, "polygon": [[1, 2], [5, 2], [5, 8], [1, 8]]}
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "map.png"
            Image.new("L", (80, 60), 255).save(image_path)
            result = request_ocr(
                image_path,
                "http://ocr",
                post=fake_post,
                max_direct_pixels=1,
                tile_size=50,
            )

        self.assertEqual(len(calls), 5)
        self.assertEqual(result["preprocessing"], "tiled")
        self.assertEqual(result["tile_count"], 4)


if __name__ == "__main__":
    unittest.main()
