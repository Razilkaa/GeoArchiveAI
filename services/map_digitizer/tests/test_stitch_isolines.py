from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.map_digitizer.stitch_isolines import propagate_values_by_tangent, read_isoline_labels


class StitchIsolinesTest(unittest.TestCase):
    def test_large_strict_small_band_label_is_kept_but_picket_is_not(self):
        payload = {
            "items": [
                {
                    "id": 1,
                    "font_band": "big",
                    "bbox": [10, 10, 90, 70],
                    "reading": {"readable": True, "text": "1.40"},
                },
                {
                    "id": 2,
                    "font_band": "small",
                    "bbox": [100, 10, 90, 90],
                    "reading": {"readable": True, "text": "1.60"},
                },
                {
                    "id": 3,
                    "font_band": "small",
                    "bbox": [200, 10, 48, 48],
                    "reading": {"readable": True, "text": "2.00"},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "readings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            labels = read_isoline_labels(path, scale=0.25)
        self.assertEqual([label["id"] for label in labels], [1, 2])

    def test_value_propagates_to_collinear_gap_only(self):
        def feature(identifier, coordinates, value=None):
            return {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "id": identifier,
                    "value": value,
                    "value_source": "nearest_label" if value is not None else "unassigned",
                    "value_conflict": False,
                    "length_px": 100.0,
                    "work_accepted": value is not None,
                },
            }

        features = [
            feature(1, [[0, 0], [25, 0], [50, 0], [75, 0], [100, 0]], 1.4),
            feature(2, [[125, 0], [150, 0], [175, 0], [200, 0], [225, 0]]),
            feature(3, [[0, 60], [25, 60], [50, 60], [75, 60], [100, 60]]),
        ]
        propagated = propagate_values_by_tangent(features)
        self.assertEqual(propagated, 1)
        self.assertEqual(features[1]["properties"]["value"], 1.4)
        self.assertIsNone(features[2]["properties"]["value"])


if __name__ == "__main__":
    unittest.main()
