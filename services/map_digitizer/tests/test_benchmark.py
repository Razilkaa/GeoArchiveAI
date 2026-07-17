from __future__ import annotations

import unittest

from services.map_digitizer.benchmark import render_markdown, summarize_results


class BenchmarkTest(unittest.TestCase):
    def test_summarizes_status_latency_and_topology(self):
        result = {
            "version": 2,
            "source": "map.jpg",
            "status": "accepted",
            "total_latency_s": 12.0,
            "stages": {
                "assignment": {"metrics": {"contour_interval_km": 0.1}},
                "reconstruction": {
                    "metrics": {
                        "reconstruction_mode": "sparse_labels_trace_guided",
                        "grid_quality": {"constraint_p95_abs_error_m": 3.0},
                        "topology": {"levels": 4, "closed_segments": 1, "crossing_pairs": 0},
                    }
                },
            },
            "quality": {
                "reasons": ["low_label_surface_agreement"],
                "direct_contour_conflict_rate": 0.2,
                "label_surface_agreement": 0.8,
            },
        }
        summary = summarize_results([result])
        self.assertEqual(summary["status_counts"], {"accepted": 1})
        self.assertEqual(summary["pipeline_version_counts"], {"2": 1})
        self.assertEqual(summary["zero_crossing_rate"], 1.0)
        self.assertEqual(
            summary["review_reason_counts"], {"low_label_surface_agreement": 1}
        )
        self.assertEqual(summary["cases"][0]["direct_conflict_rate"], 0.2)
        self.assertIn("map.jpg", render_markdown(summary))

    def test_deduplicates_repeated_source_and_prefers_adaptive_result(self):
        legacy = {
            "source": "map.jpg",
            "source_sha256": "same",
            "status": "accepted",
            "stages": {"reconstruction": {"metrics": {}}},
        }
        adaptive = {
            "source": "map.jpg",
            "source_sha256": "same",
            "status": "review",
            "version": 1,
            "stages": {
                "reconstruction": {
                    "metrics": {
                        "reconstruction_mode": "sparse_labels_trace_guided",
                        "topology": {"crossing_pairs": 0},
                    }
                }
            },
        }
        summary = summarize_results([legacy, adaptive])
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["duplicate_manifests"], 1)
        self.assertEqual(summary["cases"][0]["status"], "review")


if __name__ == "__main__":
    unittest.main()
