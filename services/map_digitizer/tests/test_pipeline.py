from __future__ import annotations

import unittest

from services.map_digitizer.pipeline import quality_decision, select_reconstruction_mode


class PipelineQualityTest(unittest.TestCase):
    def test_selects_reconstruction_mode_from_profile_measurement_density(self):
        self.assertEqual(
            select_reconstruction_mode({"profile_measurements": 20}),
            "dense_profile_measurements",
        )
        self.assertEqual(
            select_reconstruction_mode({"profile_measurements": 0}),
            "sparse_labels_trace_guided",
        )

    def test_accepts_topology_safe_low_error_surface(self):
        assignment = {"contour_interval_km": 0.1, "confident_polylines": 5}
        reconstruction = {
            "grid_quality": {
                "constraint_p95_abs_error_m": 20.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0},
        }
        self.assertEqual(quality_decision(assignment, reconstruction)["status"], "accepted")

    def test_flags_crossings_and_large_constraint_error(self):
        assignment = {"contour_interval_km": 0.025, "confident_polylines": 2}
        reconstruction = {
            "grid_quality": {
                "constraint_p95_abs_error_m": 20.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 1},
        }
        decision = quality_decision(assignment, reconstruction)
        self.assertEqual(decision["status"], "review")
        self.assertEqual(len(decision["reasons"]), 3)

    def test_flags_excessive_closed_contours(self):
        assignment = {"contour_interval_km": 0.025, "confident_polylines": 5}
        reconstruction = {
            "grid_quality": {
                "constraint_p95_abs_error_m": 2.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0, "levels": 5, "closed_segments": 12},
        }
        decision = quality_decision(assignment, reconstruction)
        self.assertIn("excessive_closed_contours", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
