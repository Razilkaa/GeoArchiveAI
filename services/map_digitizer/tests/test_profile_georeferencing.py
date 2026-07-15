from __future__ import annotations

import math
import unittest

import numpy as np

from research.map_digitization.tracing.detect_profile_lines import (
    Segment,
    angle_distance,
    consolidate_segments,
    segments_match,
)
from research.map_digitization.tracing.georeference_profiles_23 import (
    solve_similarity,
    transform_points,
)
from research.map_digitization.tracing.match_profile_labels import point_segment_distance


class ProfileDetectionTest(unittest.TestCase):
    def test_unoriented_angle_distance_wraps_at_180(self):
        self.assertAlmostEqual(angle_distance(179.0, 1.0), 2.0)

    def test_consolidates_collinear_hough_fragments(self):
        first = Segment((0, 10), (100, 10), 0, 10, 100)
        second = Segment((90, 11), (210, 11), 0, 11, 120)
        self.assertTrue(segments_match(first, second))

        lines = consolidate_segments([first, second])

        self.assertEqual(len(lines), 1)
        self.assertGreater(lines[0]["length_px"], 200)
        self.assertEqual(lines[0]["support"], 2)

    def test_point_to_segment_uses_finite_extent(self):
        distance = point_segment_distance(
            np.array([12.0, 4.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
        )
        self.assertAlmostEqual(distance, math.sqrt(20))


class ProfileGeoreferencingTest(unittest.TestCase):
    def test_recovers_reflected_similarity_from_three_lines(self):
        parameters = np.array([800_000.0, 7_000_000.0, math.log(12.0), math.radians(8.0)])
        pixel = {
            1: np.array([[0.0, 100.0], [1_000.0, 100.0]]),
            2: np.array([[0.0, 500.0], [1_000.0, 500.0]]),
            3: np.array([[300.0, 0.0], [300.0, 900.0]]),
        }
        world = {
            "A": transform_points(pixel[1], parameters),
            "B": transform_points(pixel[2], parameters),
            "C": transform_points(pixel[3], parameters),
        }

        solved, rms = solve_similarity(pixel, world, [(1, "A"), (2, "B"), (3, "C")])

        self.assertLess(rms, 0.01)
        self.assertAlmostEqual(math.exp(solved[2]), 12.0, places=3)
        self.assertAlmostEqual(math.degrees(solved[3]), 8.0, places=3)


if __name__ == "__main__":
    unittest.main()
