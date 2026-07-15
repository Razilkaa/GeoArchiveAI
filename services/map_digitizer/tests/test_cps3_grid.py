from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyproj import CRS

from services.map_digitizer.export_cps3_grid import Grid, NULL_VALUE, write_cps3


class Cps3GridTest(unittest.TestCase):
    def test_writes_all_nodes_in_cps3_column_major_top_down_order(self):
        grid = Grid(
            x=np.array([100.0, 200.0]),
            y=np.array([1000.0, 1100.0, 1200.0]),
            z=np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]),
            crs=CRS.from_epsg(28481),
        )

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "surface.cps3"
            write_cps3(grid, output, "test_surface")
            lines = output.read_text(encoding="ascii").splitlines()

        self.assertIn("FSNROW 3 2", lines)
        self.assertIn("FSXINC 100.000 100.000", lines)
        marker = lines.index("->test_surface")
        values = [float(item) for line in lines[marker + 1 :] for item in line.split()]
        self.assertEqual(len(values), 6)
        self.assertEqual(values[:3], [5.0, 3.0, 1.0])
        self.assertEqual(values[3], 6.0)
        self.assertEqual(values[4], NULL_VALUE)
        self.assertEqual(values[5], 2.0)


if __name__ == "__main__":
    unittest.main()
