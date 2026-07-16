from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.map_digitizer.assign_contour_values import dominant_value_band, normalized_label, run


class AssignContourValuesTest(unittest.TestCase):
    def test_normalizes_only_contour_interval_values(self):
        self.assertEqual(normalized_label("-3,21", 0.2, 0.06), -3.2)
        self.assertIsNone(normalized_label("-3.11", 0.2, 0.06))
        self.assertIsNone(normalized_label("91501", 0.2, 0.06))

    def test_normalizes_metre_labels_and_rejects_profile_number_band(self):
        self.assertEqual(normalized_label("-1400", 0.1, 0.06), -1.4)
        values = [-0.8, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -7.901, -7.915]
        low, high = dominant_value_band(values, 0.1)
        self.assertLess(low, -2.0)
        self.assertGreater(high, -0.8)
        self.assertGreater(low, -3.0)

    def test_scales_ocr_coordinates_to_trace_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            isolines = root / "isolines.json"
            isolines.write_text(
                json.dumps({"polylines_xy": [[[45, 10], [55, 90]]]}), encoding="utf-8"
            )
            readings = root / "readings.jsonl"
            readings.write_text(
                json.dumps(
                    {
                        "zone": "map_body",
                        "quad": [[96, 96], [104, 96], [104, 104], [96, 104]],
                        "values": [{"text": "-3.2"}],
                        "unreadable": False,
                    }
                ),
                encoding="utf-8",
            )
            (root / "baseline_metrics.json").write_text(
                json.dumps({"sheet": "synthetic (200x200 px)"}), encoding="utf-8"
            )
            image = root / "map.png"
            Image.new("L", (100, 100), 255).save(image)

            metrics = run(isolines, readings, image, root / "out", interval=0.2)

        self.assertEqual(metrics["valued_polylines"], 1)
        self.assertEqual(metrics["ocr_to_trace_scale"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
