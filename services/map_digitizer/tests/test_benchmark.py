from __future__ import annotations

import unittest

from services.map_digitizer.benchmark import render_markdown, summarize_results


class BenchmarkTest(unittest.TestCase):
    def test_summarizes_status_latency_and_topology(self):
        result = {
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
            "quality": {"reasons": []},
        }
        summary = summarize_results([result])
        self.assertEqual(summary["status_counts"], {"accepted": 1})
        self.assertEqual(summary["zero_crossing_rate"], 1.0)
        self.assertIn("map.jpg", render_markdown(summary))


if __name__ == "__main__":
    unittest.main()
