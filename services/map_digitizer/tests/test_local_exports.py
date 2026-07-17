from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from services.map_digitizer.local_exports import materialize_local_exports


class LocalExportsTest(unittest.TestCase):
    def test_materializes_clean_preview_and_local_cps3(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            grid_path = root / "surface.npz"
            np.savez_compressed(
                grid_path,
                x=np.array([0.0, 10.0, 20.0]),
                y=np.array([0.0, 10.0, 20.0]),
                z=np.array(
                    [
                        [-1000.0, -1010.0, -1020.0],
                        [-1010.0, -1020.0, -1030.0],
                        [-1020.0, -1030.0, -1040.0],
                    ]
                ),
            )
            contours_path = root / "contours.geojson"
            gpd.GeoDataFrame(
                [{"value_m": -1020.0, "geometry": LineString([(0, 20), (20, 0)])}],
                geometry="geometry",
                crs="EPSG:3857",
            ).to_file(contours_path, driver="GeoJSON")

            files = materialize_local_exports(grid_path, contours_path, root / "out")

            self.assertTrue(Path(files["surface_clean_preview"]).is_file())
            cps3 = Path(files["local_cps3"]).read_text(encoding="ascii")
            self.assertIn("LOCAL_PIXEL_COORDINATES (not georeferenced)", cps3)
            self.assertIn("FSNROW 3 3", cps3)


if __name__ == "__main__":
    unittest.main()
