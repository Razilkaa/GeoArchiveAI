"""Conservative topology cleanup for reconstructed contour artifacts."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon


def remove_unsupported_closed_contours(
    contours: gpd.GeoDataFrame,
    source_contours: gpd.GeoDataFrame,
    *,
    interval_m: float,
    support_distance: float,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Remove isolated closures with neither traced nor nested support."""
    if contours.empty or "closed" not in contours:
        return contours, {"removed_unsupported_closed": 0, "removed_ids": []}

    rings: dict[int, Polygon] = {}
    for index, row in contours[contours["closed"] == True].iterrows():
        try:
            polygon = Polygon(row.geometry)
            if polygon.is_valid and not polygon.is_empty:
                rings[int(index)] = polygon
        except (TypeError, ValueError):
            continue

    source_values = (
        source_contours["value_m"]
        if "value_m" in source_contours
        else source_contours["value_km"].astype(float) * 1000.0
    )
    supported: set[int] = set()
    for index, polygon in rings.items():
        row = contours.loc[index]
        same_level = source_contours[
            (source_values - float(row["value_m"])).abs() <= max(1.0, interval_m * 0.05)
        ]
        if not same_level.empty and float(same_level.distance(row.geometry).min()) <= support_distance:
            supported.add(index)
            continue
        for other_index, other_polygon in rings.items():
            if other_index == index:
                continue
            other_value = float(contours.loc[other_index, "value_m"])
            if abs(abs(other_value - float(row["value_m"])) - interval_m) > interval_m * 0.1:
                continue
            if polygon.contains(other_polygon.representative_point()) or other_polygon.contains(
                polygon.representative_point()
            ):
                supported.add(index)
                break

    removed = sorted(set(rings) - supported)
    cleaned = contours.drop(index=removed).reset_index(drop=True)
    return cleaned, {
        "removed_unsupported_closed": len(removed),
        "removed_ids": removed,
        "retained_closed": int(cleaned["closed"].sum()) if "closed" in cleaned else 0,
    }
