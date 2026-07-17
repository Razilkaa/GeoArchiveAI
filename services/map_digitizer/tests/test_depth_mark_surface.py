from __future__ import annotations

import unittest

import numpy as np
from shapely.geometry import LineString

from services.map_digitizer.depth_mark_surface import extract_measurements, fuse_measurement_sources
from services.map_digitizer.trace_guided_surface import prune_crossing_constraints


class DepthMarkSurfaceTest(unittest.TestCase):
    def test_reads_metre_contour_labels_as_kilometres(self):
        readings = [
            {
                "zone": "map_body",
                "quad": [[0, 0], [2, 0], [2, 2], [0, 2]],
                "values": [{"text": "-1400"}],
                "unreadable": False,
            }
        ]
        marks, labels = extract_measurements(
            readings, scale_x=1.0, scale_y=1.0, contour_interval=0.1
        )
        self.assertEqual(len(marks), 0)
        self.assertAlmostEqual(labels[0, 2], 1.4)

    def test_paddle_anchors_replace_overlapping_vlm_marks(self):
        paddle = np.array(
            [[index * 20.0, 0.0, 2.0 + index * 0.01] for index in range(12)]
        )
        vlm = paddle.copy()
        vlm[:, 0] += 2.0
        supplemental = np.array(
            [[500.0 + index * 20.0, 0.0, 2.3 + index * 0.01] for index in range(12)]
        )

        fused, metrics = fuse_measurement_sources(
            [("paddle", paddle), ("vlm", np.vstack([vlm, supplemental]))],
            radius=5.0,
        )

        self.assertEqual(metrics["mode"], "paddle_anchor_vlm_fill")
        self.assertEqual(metrics["overlapping_pairs"], 12)
        self.assertEqual(metrics["agreement_rate"], 1.0)
        self.assertEqual(len(fused), 24)

    def test_prunes_weaker_different_level_crossing(self):
        assignments = [
            {
                "id": 1, "accepted": True, "value_km": -2.0,
                "direct_label": True, "residual_km": 0.1, "spread_km": 0.1,
                "coverage": 1.0, "geometry": LineString([(0, 0), (2, 2)]),
            },
            {
                "id": 2, "accepted": True, "value_km": -2.2,
                "direct_label": False, "residual_km": 0.05, "spread_km": 0.05,
                "coverage": 1.0, "geometry": LineString([(0, 2), (2, 0)]),
            },
        ]

        removed = prune_crossing_constraints(assignments)

        self.assertEqual(removed, [2])
        self.assertTrue(assignments[0]["accepted"])
        self.assertFalse(assignments[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
