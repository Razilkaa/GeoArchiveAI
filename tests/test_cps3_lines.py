from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString

from services.map_digitizer.export_cps3_lines import NULL_VALUE, write_cps3_lines


def test_cps3_lines_preserves_subsets_values_and_nulls(tmp_path: Path) -> None:
    frame = gpd.GeoDataFrame(
        {
            "name": ["valued contour", None],
            "value_m": [-3200.0, np.nan],
        },
        geometry=[
            LineString([(100.0, 200.0), (110.0, 210.0)]),
            MultiLineString([
                [(300.0, 400.0), (310.0, 410.0)],
                [(500.0, 600.0), (510.0, 610.0)],
            ]),
        ],
        crs="EPSG:28479",
    )

    path = write_cps3_lines(frame, tmp_path / "contours.cps3")
    text = path.read_text(encoding="ascii")

    assert "! CRS: EPSG:28479" in text
    assert "->valued_contour" in text
    assert "100.000 200.000 -3200.000000" in text
    assert sum(line.startswith("->") for line in text.splitlines()) == 3
    assert f"{NULL_VALUE:.6E}" in text
