from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.map_digitizer.georeference_grid import fit_affine, georeference_grid, transform_xy


class GeoreferenceGridTest(unittest.TestCase):
    def test_fits_affine_with_independent_check_point(self):
        controls = [
            {"pixel": [0, 0], "map": [1000, 2000]},
            {"pixel": [10, 0], "map": [1020, 2000]},
            {"pixel": [0, 10], "map": [1000, 1970]},
            {"pixel": [10, 10], "map": [1020, 1970]},
        ]
        matrix, quality = fit_affine(controls)
        transformed = transform_xy(np.asarray([[5.0, 5.0]]), matrix)[0]
        np.testing.assert_allclose(transformed, [1010.0, 1985.0], atol=1e-8)
        self.assertLess(quality["p95_m"], 1e-8)

    def test_exports_regular_cps_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "surface.npz"
            np.savez_compressed(
                source,
                x=np.asarray([0.0, 1.0, 2.0]),
                y=np.asarray([0.0, 1.0, 2.0]),
                z=np.asarray([[-100.0, -101.0, -102.0]] * 3),
            )
            controls = [
                {"pixel": [0, 0], "map": [1000, 2000]},
                {"pixel": [2, 0], "map": [1020, 2000]},
                {"pixel": [0, 2], "map": [1000, 1980]},
                {"pixel": [2, 2], "map": [1020, 1980]},
            ]
            result = georeference_grid(
                source, controls, root / "out", target_crs="EPSG:28481", cell_size=10.0
            )
            cps = Path(result["files"]["cps3"]).read_text(encoding="ascii")
        self.assertEqual(result["status"], "accepted")
        self.assertIn("FSNROW 3 3", cps)


if __name__ == "__main__":
    unittest.main()
