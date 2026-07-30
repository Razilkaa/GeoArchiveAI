"""CPS-3 ASCII line-set export for editable contour geometry."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString


NULL_VALUE = 1.0e30
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _line_parts(geometry: object) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _subset_name(raw: object, fallback: str) -> str:
    value = _SAFE_NAME.sub("_", str(raw or fallback)).strip("_.")
    return value[:72] or fallback


def write_cps3_lines(
    frame: gpd.GeoDataFrame,
    output_path: str | Path,
    *,
    value_field: str = "value_m",
    crs_label: str | None = None,
) -> Path:
    """Write LineString features as CPS-3 subsets with X/Y/Z vertices."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crs_text = crs_label or (str(frame.crs) if frame.crs else "UNKNOWN")
    rows = [
        "! CPS-3 ASCII line set",
        f"! CRS: {crs_text}",
        f"! Z null value: {NULL_VALUE:.6E}",
        "! Each -> marker starts one editable polyline subset.",
    ]
    line_index = 0
    for feature_index, feature in frame.iterrows():
        raw_value = feature.get(value_field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = NULL_VALUE
        if not math.isfinite(value):
            value = NULL_VALUE
        raw_name = feature.get("name") or feature.get("id")
        for part_index, geometry in enumerate(_line_parts(feature.geometry)):
            if geometry.is_empty or len(geometry.coords) < 2:
                continue
            line_index += 1
            fallback = f"line_{line_index:06d}"
            suffix = f"_{part_index + 1}" if part_index else ""
            rows.append(f"->{_subset_name(raw_name, fallback)}{suffix}")
            for coordinate in geometry.coords:
                z = value
                if len(coordinate) >= 3 and math.isfinite(float(coordinate[2])):
                    z = float(coordinate[2])
                z_text = f"{z:.6E}" if z == NULL_VALUE else f"{z:.6f}"
                rows.append(
                    f"{float(coordinate[0]):.3f} "
                    f"{float(coordinate[1]):.3f} "
                    f"{z_text}"
                )
    output_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    return output_path
