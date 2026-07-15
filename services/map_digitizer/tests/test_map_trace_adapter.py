from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.map_digitizer.map_trace_adapter import build_map_result


class MapTraceAdapterTest(unittest.TestCase):
    def test_marks_experimental_pixel_trace_for_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "combined_23.png").write_bytes(b"png")
            (root / "valued_isolines_23.json").write_text(
                json.dumps(
                    {
                        "n_polylines": 10,
                        "n_valued": 3,
                        "contour_step_km": 0.2,
                        "profile_leaks": [0, 1],
                    }
                ),
                encoding="utf-8",
            )
            (root / "crosscheck_23.json").write_text(
                json.dumps(
                    {
                        "verdicts": {"2": "main", "3": "flagged", "4": "combo"},
                        "from_surface": {"4": -2.0},
                        "structures": [],
                    }
                ),
                encoding="utf-8",
            )
            features = [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                    "properties": {
                        "id": index,
                        "kind": "seismic_profile" if index < 2 else "isoline",
                        "source": "label" if index in {2, 3, 4} else "untraced_value",
                        "value_km": -2.0 if index in {2, 3, 4} else None,
                        "closed": False,
                    },
                }
                for index in range(10)
            ]
            features[2]["geometry"]["coordinates"] = [[0, 0], [2, 2]]
            features[3]["geometry"]["coordinates"] = [[0, 2], [2, 0]]
            (root / "isolines_23.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
            )

            output = root / "result.json"
            payload = build_map_result("demo", root, output)

            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["quality_status"], "review")
            self.assertFalse(payload["metrics"]["georeferenced"])
            self.assertEqual(payload["metrics"]["profile_leak_rate"], 0.2)
            self.assertEqual(payload["metrics"]["crosscheck_within_one_step_rate"], 0.5)
            self.assertEqual(payload["metrics"]["closed_contours"], 0)
            self.assertEqual(payload["metrics"]["crossing_isoline_pairs"], 1)
            self.assertEqual(payload["metrics"]["crossing_pair_ids"], [[2, 3]])
            self.assertTrue(output.exists())

    def test_exposes_provisional_georeferenced_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "combined_23.png").write_bytes(b"png")
            (root / "georeference_preview.png").write_bytes(b"png")
            (root / "sheet_23_provisional.gpkg").write_bytes(b"gpkg")
            (root / "sheet_23_horizon_k_gk42_21n.png").write_bytes(b"png")
            (root / "sheet_23_horizon_k_gk42_21n.cps3").write_bytes(b"cps3")
            (root / "sheet_23_horizon_k_gk42_19n.cps3").write_bytes(b"cps3")
            (root / "georeference_qc.json").write_text(
                json.dumps(
                    {
                        "status": "review",
                        "crs": "EPSG:2509",
                        "selected_solution": {"rms_m": 73.48},
                        "ambiguity_ratio": 1.278,
                    }
                ),
                encoding="utf-8",
            )
            (root / "valued_isolines_23.json").write_text(
                json.dumps({"n_polylines": 1, "n_valued": 1, "profile_leaks": []}),
                encoding="utf-8",
            )
            (root / "crosscheck_23.json").write_text(
                json.dumps({"verdicts": {"0": "main"}, "from_surface": {}, "structures": []}),
                encoding="utf-8",
            )
            (root / "isolines_23.geojson").write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                                "properties": {"id": 0, "kind": "isoline", "source": "label"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_map_result("demo", root, root / "result.json")

            self.assertTrue(payload["metrics"]["georeferenced"])
            self.assertEqual(payload["metrics"]["coordinate_system"], "EPSG:2509")
            self.assertEqual(payload["metrics"]["georeference_rms_m"], 73.48)
            self.assertEqual(
                [item["name"] for item in payload["artifacts"]],
                [
                    "surface_grid_preview",
                    "surface_grid_21n",
                    "surface_grid_19n",
                    "georeferenced_preview",
                    "geopackage",
                ],
            )


if __name__ == "__main__":
    unittest.main()
