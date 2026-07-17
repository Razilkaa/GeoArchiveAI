from __future__ import annotations

import unittest

from services.map_digitizer.pipeline import (
    assess_ocr_eligibility,
    digitized_contours_path,
    quality_decision,
    select_reconstruction_mode,
    insufficient_assignment_support,
    insufficient_reconstruction_support,
)


class PipelineQualityTest(unittest.TestCase):
    def test_prefers_source_geometry_for_digitized_contour_export(self):
        self.assertEqual(
            digitized_contours_path(
                {
                    "files": {
                        "digitized_contours": "source.geojson",
                        "final_contours": "synthetic.geojson",
                    }
                }
            ),
            "source.geojson",
        )
        self.assertEqual(
            digitized_contours_path({"files": {"final_contours": "fallback.geojson"}}),
            "fallback.geojson",
        )

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
            select_reconstruction_mode({"profile_measurements": 80}),
            "dense_profile_measurements",
        )
        self.assertEqual(
            select_reconstruction_mode({"profile_measurements": 79}),
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

    def test_flags_implausibly_dense_level_family(self):
        assignment = {"contour_interval_km": 0.01, "confident_polylines": 5}
        reconstruction = {
            "accepted_traces": 40,
            "grid_quality": {
                "constraint_p95_abs_error_m": 2.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0, "levels": 95, "closed_segments": 164},
        }
        decision = quality_decision(assignment, reconstruction)
        self.assertIn("excessive_contour_levels", decision["reasons"])
        self.assertIn("excessive_closed_contours", decision["reasons"])

    def test_sparse_surface_without_profile_network_requires_review(self):
        assignment = {
            "contour_interval_km": 0.2,
            "confident_polylines": 6,
            "profile_id_labels": 5,
        }
        reconstruction = {
            "reconstruction_mode": "sparse_labels_trace_guided",
            "accepted_traces": 30,
            "grid_quality": {
                "constraint_p95_abs_error_m": 5.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0, "levels": 5, "closed_segments": 2},
        }
        decision = quality_decision(assignment, reconstruction)
        self.assertIn("weak_profile_network_evidence", decision["reasons"])

    def test_sparse_surface_always_requires_semantic_review(self):
        assignment = {
            "contour_interval_km": 0.1,
            "confident_polylines": 20,
            "conflicting_polylines": 1,
            "profile_id_labels": 30,
        }
        reconstruction = {
            "reconstruction_mode": "sparse_labels_trace_guided",
            "accepted_traces": 100,
            "grid_quality": {
                "constraint_p95_abs_error_m": 2.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0, "levels": 15, "closed_segments": 2},
        }

        decision = quality_decision(assignment, reconstruction)

        self.assertEqual(decision["status"], "review")
        self.assertIn("sparse_reconstruction_requires_review", decision["reasons"])

    def test_high_direct_contour_conflict_rate_requires_review(self):
        assignment = {
            "contour_interval_km": 0.1,
            "confident_polylines": 8,
            "conflicting_polylines": 6,
            "profile_id_labels": 20,
        }
        reconstruction = {
            "reconstruction_mode": "sparse_labels_trace_guided",
            "accepted_traces": 50,
            "grid_quality": {
                "constraint_p95_abs_error_m": 5.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0, "levels": 10, "closed_segments": 2},
        }

        decision = quality_decision(assignment, reconstruction)

        self.assertEqual(decision["status"], "review")
        self.assertIn("high_direct_contour_conflict_rate", decision["reasons"])
        self.assertEqual(decision["direct_contour_conflict_rate"], 0.4286)

    def test_low_label_surface_agreement_requires_review(self):
        assignment = {
            "contour_interval_km": 0.1,
            "confident_polylines": 8,
            "conflicting_polylines": 1,
            "profile_id_labels": 20,
        }
        reconstruction = {
            "reconstruction_mode": "sparse_labels_trace_guided",
            "accepted_traces": 50,
            "grid_quality": {
                "constraint_p95_abs_error_m": 5.0,
                "value_range_preserved": True,
            },
            "topology": {"crossing_pairs": 0, "levels": 10, "closed_segments": 2},
            "preliminary_surface": {
                "label_crosscheck": {"compared": 23, "within_one_interval_rate": 0.43}
            },
        }

        decision = quality_decision(assignment, reconstruction)

        self.assertEqual(decision["status"], "review")
        self.assertIn("low_label_surface_agreement", decision["reasons"])
        self.assertEqual(decision["label_surface_agreement"], 0.43)

    def test_recognizes_expected_sparse_reconstruction_failure(self):
        self.assertTrue(
            insufficient_reconstruction_support(
                ValueError("Only 2 traced contours passed QC")
            )
        )
        self.assertFalse(insufficient_reconstruction_support(ValueError("broken grid shape")))

    def test_recognizes_all_expected_insufficient_support_failures(self):
        self.assertTrue(
            insufficient_assignment_support(
                ValueError("Cannot infer contour interval from fewer than three labels")
            )
        )
        self.assertTrue(
            insufficient_reconstruction_support(
                ValueError("Only 2 trusted contours; at least 3 required")
            )
        )
        self.assertTrue(
            insufficient_reconstruction_support(
                ValueError("Only 3 valid depth constraints; at least 5 are required")
            )
        )


if __name__ == "__main__":
    unittest.main()
