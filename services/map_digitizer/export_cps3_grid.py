"""Build a Petrel-ready CPS-3 ASCII surface from valued contour lines."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from pyproj import CRS
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import label as label_components
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from shapely.geometry import LineString
from skimage.measure import find_contours

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NULL_VALUE = 1.0e30


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    crs: CRS


def sample_contours(contours: gpd.GeoDataFrame, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    values: list[float] = []
    for row in contours.itertuples():
        geometry = row.geometry
        if geometry.geom_type == "Point":
            sampled = [geometry]
        elif geometry.geom_type == "LineString":
            count = max(2, math.ceil(geometry.length / spacing) + 1)
            sampled = [geometry.interpolate(float(distance)) for distance in np.linspace(0.0, geometry.length, count)]
        else:
            continue
        for point in sampled:
            points.append((point.x, point.y))
            values.append(float(row.value_km) * 1_000.0)

    coordinates = np.asarray(points)
    z_values = np.asarray(values)
    rounded = np.round(coordinates, 3)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    if len(unique) != len(coordinates):
        merged = np.array([np.median(z_values[inverse == index]) for index in range(len(unique))])
        return unique, merged
    return coordinates, z_values


def sample_weighted_contours(
    contours: gpd.GeoDataFrame, spacing: float, default_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    values: list[float] = []
    weights: list[float] = []
    for row in contours.itertuples():
        geometry = row.geometry
        if geometry.geom_type == "Point":
            sampled = [geometry]
        elif geometry.geom_type == "LineString":
            count = max(2, math.ceil(geometry.length / spacing) + 1)
            sampled = [
                geometry.interpolate(float(distance))
                for distance in np.linspace(0.0, geometry.length, count)
            ]
        else:
            continue
        weight = float(getattr(row, "constraint_weight", default_weight))
        for point in sampled:
            points.append((point.x, point.y))
            values.append(float(row.value_km) * 1000.0)
            weights.append(weight)
    return np.asarray(points), np.asarray(values), np.asarray(weights)


def build_grid(
    contours: gpd.GeoDataFrame,
    *,
    cell_size: float = 250.0,
    blanking_distance: float = 4_000.0,
) -> Grid:
    points, values = sample_contours(contours, cell_size)
    min_x, min_y = np.floor(points.min(axis=0) / cell_size) * cell_size
    max_x, max_y = np.ceil(points.max(axis=0) / cell_size) * cell_size
    x = np.arange(min_x, max_x + cell_size * 0.5, cell_size)
    y = np.arange(min_y, max_y + cell_size * 0.5, cell_size)
    grid_x, grid_y = np.meshgrid(x, y)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
    z = np.asarray(interpolator(nodes)).reshape(grid_x.shape)
    nearest_distance, _ = cKDTree(points).query(nodes, k=1)
    z.reshape(-1)[nearest_distance > blanking_distance] = np.nan
    return Grid(x=x, y=y, z=z, crs=CRS.from_user_input(contours.crs))


def build_harmonic_grid(
    contours: gpd.GeoDataFrame,
    *,
    cell_size: float = 250.0,
    blanking_distance: float = 4_000.0,
    constraint_weight: float = 20.0,
    support_contours: gpd.GeoDataFrame | None = None,
) -> tuple[Grid, dict]:
    """Interpolate between regularized contour constraints without overshoot.

    A discrete Laplace solution is preferable to unconstrained RBF/triangulation
    here: it honours the digitized contours, remains inside their value range,
    and produces one continuous surface whose derived contours cannot cross.
    """
    points, values, point_weights = sample_weighted_contours(
        contours, min(cell_size * 0.5, 125.0), constraint_weight
    )
    support = support_contours if support_contours is not None and len(support_contours) else contours
    support_points, _ = sample_contours(
        support.assign(value_km=0.0), min(cell_size * 0.5, 125.0)
    )
    min_x, min_y = np.floor(support_points.min(axis=0) / cell_size) * cell_size
    max_x, max_y = np.ceil(support_points.max(axis=0) / cell_size) * cell_size
    x = np.arange(min_x, max_x + cell_size * 0.5, cell_size)
    y = np.arange(min_y, max_y + cell_size * 0.5, cell_size)
    grid_x, grid_y = np.meshgrid(x, y)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    distance, _ = cKDTree(support_points).query(nodes, k=1)
    if len(support_points) >= 3:
        inside_hull = Delaunay(support_points).find_simplex(nodes) >= 0
    else:
        inside_hull = np.ones(len(nodes), dtype=bool)
    mask = (distance <= blanking_distance) & inside_hull
    mask = mask.reshape(grid_x.shape)

    # Burn every source contour into grid cells. Conflicting values in one cell
    # are retained as a QC signal and resolved by a median, never silently.
    col = np.clip(np.rint((points[:, 0] - x[0]) / cell_size).astype(int), 0, len(x) - 1)
    row = np.clip(np.rint((points[:, 1] - y[0]) / cell_size).astype(int), 0, len(y) - 1)
    cell_values: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for r, c, value, weight in zip(row, col, values, point_weights):
        if mask[r, c]:
            cell_values.setdefault((int(r), int(c)), []).append((float(value), float(weight)))
    fixed = {
        cell: float(np.average([item[0] for item in items], weights=[item[1] for item in items]))
        for cell, items in cell_values.items()
    }
    fixed_weights = {
        cell: float(max(item[1] for item in items)) for cell, items in cell_values.items()
    }
    conflict_cells = sum(
        max(item[0] for item in items) - min(item[0] for item in items) > 1.0
        for items in cell_values.values()
    )

    # Keep only connected support regions that contain at least one trusted
    # value. This expands to unlabelled traced margins without inventing a
    # surface for detached, unconstrained map decoration.
    component_map, component_count = label_components(mask)
    constrained_components = {
        int(component_map[r, c]) for r, c in fixed if component_map[r, c] > 0
    }
    if component_count:
        mask = np.isin(component_map, list(constrained_components))
        fixed = {cell: value for cell, value in fixed.items() if mask[cell]}
        fixed_weights = {cell: value for cell, value in fixed_weights.items() if mask[cell]}

    active = [tuple(item) for item in np.argwhere(mask)]
    active_index = {cell: index for index, cell in enumerate(active)}
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_data: list[float] = []
    rhs = np.zeros(len(active), dtype=float)
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for index, (r, c) in enumerate(active):
        degree = 0
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= len(y) or nc < 0 or nc >= len(x) or not mask[nr, nc]:
                continue
            degree += 1
            neighbour = (nr, nc)
            matrix_rows.append(index)
            matrix_cols.append(active_index[neighbour])
            matrix_data.append(-1.0)
        diagonal = float(max(degree, 1))
        if (r, c) in fixed:
            weight = fixed_weights[(r, c)]
            diagonal += weight
            rhs[index] += weight * fixed[(r, c)]
        matrix_rows.append(index)
        matrix_cols.append(index)
        matrix_data.append(diagonal)

    z = np.full(grid_x.shape, np.nan, dtype=float)
    if active:
        matrix = csr_matrix((matrix_data, (matrix_rows, matrix_cols)), shape=(len(active), len(active)))
        solution = spsolve(matrix, rhs)
        for (r, c), value in zip(active, solution):
            z[r, c] = value

    sampled = z[row, col]
    residual = sampled - values
    residual = residual[np.isfinite(residual)]
    quality = {
        "method": "harmonic_regularized_contours",
        "constraint_points": int(len(points)),
        "support_points": int(len(support_points)),
        "support_components": int(len(constrained_components)),
        "fixed_cells": int(len(fixed)),
        "constraint_weight": float(constraint_weight),
        "constraint_weight_range": [float(point_weights.min()), float(point_weights.max())],
        "conflicting_fixed_cells": int(conflict_cells),
        "constraint_rmse_m": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else None,
        "constraint_p95_abs_error_m": (
            float(np.percentile(np.abs(residual), 95)) if len(residual) else None
        ),
        "value_range_preserved": bool(
            np.nanmin(z) >= values.min() - 1e-6 and np.nanmax(z) <= values.max() + 1e-6
        ),
    }
    return Grid(x=x, y=y, z=z, crs=CRS.from_user_input(contours.crs)), quality


def extract_surface_contours(grid: Grid, interval: float = 200.0) -> gpd.GeoDataFrame:
    """Vectorize final grid contours; these are the production map lines."""
    finite = grid.z[np.isfinite(grid.z)]
    first = math.ceil(float(finite.min()) / interval) * interval
    last = math.floor(float(finite.max()) / interval) * interval
    levels = np.arange(first, last + interval * 0.5, interval)
    mask = np.isfinite(grid.z)
    rows = []
    dx = float(grid.x[1] - grid.x[0]) if len(grid.x) > 1 else 0.0
    dy = float(grid.y[1] - grid.y[0]) if len(grid.y) > 1 else 0.0
    for level in levels:
        for path in find_contours(grid.z, float(level), mask=mask):
            if len(path) < 4:
                continue
            coordinates = [
                (float(grid.x[0] + point[1] * dx), float(grid.y[0] + point[0] * dy))
                for point in path
            ]
            line = LineString(coordinates)
            if line.length < max(dx, dy) * 2:
                continue
            rows.append(
                {
                    "value_m": float(level),
                    "value_km": float(level / 1_000.0),
                    "closed": bool(np.linalg.norm(path[0] - path[-1]) <= 1.5),
                    "geometry": line,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=grid.crs)


def contour_topology(contours: gpd.GeoDataFrame) -> dict:
    crossings = []
    records = list(contours.itertuples())
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            if left.geometry.crosses(right.geometry):
                crossings.append([left_index, right_index])
    return {
        "segments": len(contours),
        "levels": int(contours["value_m"].nunique()) if len(contours) else 0,
        "closed_segments": int(contours["closed"].sum()) if len(contours) else 0,
        "crossing_pairs": len(crossings),
        "crossing_pair_ids": crossings,
    }


def write_cps3(grid: Grid, output: Path, name: str) -> None:
    finite = grid.z[np.isfinite(grid.z)]
    if finite.size == 0:
        raise ValueError("Grid contains no finite cells")
    dx = float(grid.x[1] - grid.x[0]) if len(grid.x) > 1 else 0.0
    dy = float(grid.y[1] - grid.y[0]) if len(grid.y) > 1 else 0.0
    lines = [
        f'! Surface: {name}',
        f'! CRS: {grid.crs.to_string()} | {grid.crs.name}',
        '! Z unit: metre; negative values are subsea depth/elevation',
        'FSASCI 0 1 "Computed" 0 1E30 0',
        'FSATTR 4 2 2 0',
        (
            f'FSLIMI {grid.x.min():.3f} {grid.x.max():.3f} '
            f'{grid.y.min():.3f} {grid.y.max():.3f} '
            f'{finite.min():.3f} {finite.max():.3f}'
        ),
        f'FSNROW {len(grid.y)} {len(grid.x)}',
        f'FSXINC {dx:.3f} {dy:.3f}',
        f'->{name}',
    ]

    # CPS-3 ordering: columns left-to-right, each column top-to-bottom.
    values = []
    for column in range(len(grid.x)):
        for row in range(len(grid.y) - 1, -1, -1):
            value = grid.z[row, column]
            values.append(NULL_VALUE if not np.isfinite(value) else float(value))
    for offset in range(0, len(values), 5):
        lines.append(" ".join(f"{value:.8E}" for value in values[offset : offset + 5]))
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_xyz(grid: Grid, output: Path) -> None:
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("X Y Z\n")
        for row, y_value in enumerate(grid.y):
            for column, x_value in enumerate(grid.x):
                z_value = grid.z[row, column]
                if np.isfinite(z_value):
                    stream.write(f"{x_value:.3f} {y_value:.3f} {z_value:.3f}\n")


def render_preview(
    grid: Grid,
    source_contours: gpd.GeoDataFrame,
    reconstructed: gpd.GeoDataFrame,
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
    masked = np.ma.masked_invalid(grid.z)
    image = ax.pcolormesh(grid.x, grid.y, masked, shading="auto", cmap="viridis_r")
    reconstructed.plot(ax=ax, color="#101010", linewidth=0.7, alpha=0.9)
    source_contours.plot(ax=ax, color="white", linewidth=0.5, alpha=0.8)
    fig.colorbar(image, ax=ax, label="Depth/elevation, m")
    ax.set_title(title)
    ax.set_xlabel("Easting, m")
    ax.set_ylabel("Northing, m")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def export_surface(
    source: Path,
    output_dir: Path,
    *,
    target_crs: str,
    source_crs: str = "EPSG:28479",
    cell_size: float = 250.0,
    blanking_distance: float = 4_000.0,
    include_inferred: bool = False,
) -> dict:
    all_contours = gpd.read_file(source, layer="isolines")
    support_contours = all_contours[all_contours["kind"].eq("isoline")].copy()
    contours = all_contours[
        all_contours["value_km"].notna()
        & all_contours["kind"].eq("isoline")
        & all_contours["verdict"].fillna("").ne("flagged")
    ].copy()
    if not include_inferred:
        contours = contours[contours["source"].eq("label")].copy()
    if contours.empty:
        raise ValueError("No trusted valued contours are available for gridding")
    contours = contours.set_crs(source_crs, allow_override=True).to_crs(target_crs)
    support_contours = support_contours.set_crs(source_crs, allow_override=True).to_crs(target_crs)
    grid, quality = build_harmonic_grid(
        contours,
        cell_size=cell_size,
        blanking_distance=blanking_distance,
        support_contours=support_contours,
    )
    reconstructed = extract_surface_contours(grid)
    quality["topology"] = contour_topology(reconstructed)

    crs = CRS.from_user_input(target_crs)
    zone = "gk42_21n" if crs.to_epsg() == 28481 else "gk42_19n"
    match = re.search(r"sheet[_-]?(\d+)", source.stem, flags=re.IGNORECASE)
    sheet_name = f"sheet_{match.group(1)}" if match else source.stem
    stem = f"{sheet_name}_horizon_k_{zone}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cps3_path = output_dir / f"{stem}.cps3"
    xyz_path = output_dir / f"{stem}.xyz"
    prj_path = output_dir / f"{stem}.prj"
    preview_path = output_dir / f"{stem}.png"
    metadata_path = output_dir / f"{stem}.json"
    contours_path = output_dir / f"{stem}_isolines.geojson"

    write_cps3(grid, cps3_path, "Horizon_K_depth_m")
    write_xyz(grid, xyz_path)
    prj_path.write_text(crs.to_wkt("WKT1_ESRI"), encoding="ascii")
    reconstructed.to_file(contours_path, driver="GeoJSON")
    render_preview(
        grid,
        contours,
        reconstructed,
        preview_path,
        f"Horizon K | reconstructed contours | {crs.name} | REVIEW",
    )

    finite = grid.z[np.isfinite(grid.z)]
    metadata = {
        "status": "review",
        "source": str(source),
        "source_crs": CRS.from_user_input(source_crs).to_string(),
        "target_crs": crs.to_string(),
        "target_crs_name": crs.name,
        "cell_size_m": cell_size,
        "blanking_distance_m": blanking_distance,
        "columns": len(grid.x),
        "rows": len(grid.y),
        "finite_cells": int(finite.size),
        "null_cells": int(grid.z.size - finite.size),
        "bounds": [float(grid.x.min()), float(grid.y.min()), float(grid.x.max()), float(grid.y.max())],
        "z_min_m": float(finite.min()),
        "z_max_m": float(finite.max()),
        "z_unit": "m",
        "input_contours": len(contours),
        "input_policy": "labels_and_inferred" if include_inferred else "trusted_labels_only",
        "quality": quality,
        "files": {
            "cps3": str(cps3_path),
            "xyz": str(xyz_path),
            "prj": str(prj_path),
            "preview": str(preview_path),
            "isolines": str(contours_path),
        },
        "warning": "Georeferencing and contour values are provisional; validate against a labelled cross-profile.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-crs", default="EPSG:28481")
    parser.add_argument("--source-crs", default="EPSG:28479")
    parser.add_argument("--cell-size", type=float, default=250.0)
    parser.add_argument("--blanking-distance", type=float, default=4_000.0)
    parser.add_argument("--include-inferred", action="store_true")
    args = parser.parse_args()
    result = export_surface(
        args.source,
        args.output_dir,
        target_crs=args.target_crs,
        source_crs=args.source_crs,
        cell_size=args.cell_size,
        blanking_distance=args.blanking_distance,
        include_inferred=args.include_inferred,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
