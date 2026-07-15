"""Build a Petrel-ready CPS-3 ASCII surface from valued contour lines."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from pyproj import CRS
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString

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
        geometry: LineString = row.geometry
        count = max(2, math.ceil(geometry.length / spacing) + 1)
        for distance in np.linspace(0.0, geometry.length, count):
            point = geometry.interpolate(float(distance))
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


def render_preview(grid: Grid, contours: gpd.GeoDataFrame, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
    masked = np.ma.masked_invalid(grid.z)
    image = ax.pcolormesh(grid.x, grid.y, masked, shading="auto", cmap="viridis_r")
    contours.plot(ax=ax, color="white", linewidth=0.45, alpha=0.7)
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
) -> dict:
    contours = gpd.read_file(source, layer="isolines")
    contours = contours[
        contours["value_km"].notna()
        & contours["kind"].eq("isoline")
        & contours["verdict"].fillna("").ne("flagged")
    ].copy()
    contours = contours.set_crs(source_crs, allow_override=True).to_crs(target_crs)
    grid = build_grid(contours, cell_size=cell_size, blanking_distance=blanking_distance)

    crs = CRS.from_user_input(target_crs)
    zone = "gk42_21n" if crs.to_epsg() == 28481 else "gk42_19n"
    stem = f"sheet_23_horizon_k_{zone}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cps3_path = output_dir / f"{stem}.cps3"
    xyz_path = output_dir / f"{stem}.xyz"
    prj_path = output_dir / f"{stem}.prj"
    preview_path = output_dir / f"{stem}.png"
    metadata_path = output_dir / f"{stem}.json"

    write_cps3(grid, cps3_path, "Horizon_K_depth_m")
    write_xyz(grid, xyz_path)
    prj_path.write_text(crs.to_wkt("WKT1_ESRI"), encoding="ascii")
    render_preview(grid, contours, preview_path, f"Horizon K grid | {crs.name} | REVIEW")

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
        "files": {
            "cps3": str(cps3_path),
            "xyz": str(xyz_path),
            "prj": str(prj_path),
            "preview": str(preview_path),
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
    args = parser.parse_args()
    result = export_surface(
        args.source,
        args.output_dir,
        target_crs=args.target_crs,
        source_crs=args.source_crs,
        cell_size=args.cell_size,
        blanking_distance=args.blanking_distance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
