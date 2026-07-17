from __future__ import annotations

import unittest

from services.map_digitizer.pipeline import (
    assess_ocr_eligibility,
    quality_decision,
    select_reconstruction_mode,
    insufficient_reconstruction_support,
)


class PipelineQualityTest(unittest.TestCase):
    def test_ocr_eligibility_requires_repeated_depth_levels(self):
        eligible = assess_ocr_eligibility(
            {
                "lines": [
                    {"text": value, "score": 0.99}
                    for value in ["-800", "-900", "-1000", "-1100", "-1200"]
                ]
            }
        )
        rejected = assess_ocr_eligibility(
            {"lines": [{"text": "91501", "score": 0.99}, {"text": "profile", "score": 0.99}]}
        )
        self.assertTrue(eligible["eligible"])
        self.assertFalse(rejected["eligible"])

    def test_ocr_eligibility_rejects_section_aspect_ratio(self):
        result = assess_ocr_eligibility(
            {
                "lines": [
                    {"text": value, "score": 0.99}
                    for value in ["-800", "-900", "-1000", "-1100", "-1200"]
                ]
            },
            (20_000, 5_000),
        )
        self.assertFalse(result["eligible"])
        self.assertIn("extreme_aspect_ratio", result["reasons"])

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

    def test_recognizes_expected_sparse_reconstruction_failure(self):
        self.assertTrue(
            insufficient_reconstruction_support(
                ValueError("Only 2 traced contours passed QC")
            )
        )
        self.assertFalse(insufficient_reconstruction_support(ValueError("broken grid shape")))


if __name__ == "__main__":
    unittest.main()
