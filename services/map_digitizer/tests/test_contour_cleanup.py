from __future__ import annotations

import unittest

import geopandas as gpd
from shapely.geometry import LineString

from services.map_digitizer.contour_cleanup import remove_unsupported_closed_contours


def ring(size: float, offset: float = 0.0) -> LineString:
    return LineString(
        [
            (offset, offset),
            (offset + size, offset),
            (offset + size, offset + size),
            (offset, offset + size),
            (offset, offset),
        ]
    )


class ContourCleanupTest(unittest.TestCase):
    def test_removes_isolated_unsupported_closure(self):
        contours = gpd.GeoDataFrame(
            [{"value_m": -1000.0, "closed": True, "geometry": ring(10)}],
            geometry="geometry",
        )
        source = gpd.GeoDataFrame(
            [{"value_km": -1.2, "geometry": LineString([(100, 0), (120, 0)])}],
            geometry="geometry",
        )
        cleaned, metrics = remove_unsupported_closed_contours(
            contours, source, interval_m=100, support_distance=5
        )
        self.assertTrue(cleaned.empty)
        self.assertEqual(metrics["removed_unsupported_closed"], 1)

    def test_keeps_directly_supported_and_nested_adjacent_closures(self):
        contours = gpd.GeoDataFrame(
            [
                {"value_m": -1000.0, "closed": True, "geometry": ring(20)},
                {"value_m": -900.0, "closed": True, "geometry": ring(10, 5)},
                {"value_m": -800.0, "closed": True, "geometry": ring(5, 40)},
            ],
            geometry="geometry",
        )
        source = gpd.GeoDataFrame(
            [{"value_km": -0.8, "geometry": ring(5, 41)}], geometry="geometry"
        )
        cleaned, metrics = remove_unsupported_closed_contours(
            contours, source, interval_m=100, support_distance=2
        )
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(metrics["removed_unsupported_closed"], 0)


if __name__ == "__main__":
    unittest.main()
