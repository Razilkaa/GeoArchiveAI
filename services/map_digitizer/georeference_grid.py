"""Georeference a pixel surface with control points and export CPS-3."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pyproj import CRS
from scipy.interpolate import LinearNDInterpolator

from services.map_digitizer.export_cps3_grid import Grid, write_cps3, write_xyz


def fit_affine(control_points: list[dict]) -> tuple[np.ndarray, dict]:
    if len(control_points) < 3:
        raise ValueError("At least three pixel-to-map control points are required")
    pixels = np.asarray([point["pixel"] for point in control_points], dtype=float)
    mapped = np.asarray([point["map"] for point in control_points], dtype=float)
    if pixels.shape != (len(control_points), 2) or mapped.shape != pixels.shape:
        raise ValueError("Each control point must contain pixel [x,y] and map [x,y]")
    design = np.column_stack([pixels, np.ones(len(pixels))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Control points are collinear")
    coefficients, _, _, _ = np.linalg.lstsq(design, mapped, rcond=None)
    matrix = coefficients.T
    predicted = design @ coefficients
    residuals = np.linalg.norm(predicted - mapped, axis=1)
    return matrix, {
        "count": len(control_points),
        "rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "median_m": float(np.median(residuals)),
        "p95_m": float(np.percentile(residuals, 95)),
        "max_m": float(residuals.max()),
        "residuals_m": residuals.tolist(),
    }


def transform_xy(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points))]) @ matrix.T


def georeference_grid(
    pixel_grid_path: Path,
    control_points: list[dict],
    output_dir: Path,
    *,
    target_crs: str,
    cell_size: float | None = None,
    name: str = "digitized_surface",
) -> dict:
    matrix, control_quality = fit_affine(control_points)
    payload = np.load(pixel_grid_path)
    source_x, source_y, source_z = payload["x"], payload["y"], payload["z"]
    source_grid_x, source_grid_y = np.meshgrid(source_x, source_y)
    finite = np.isfinite(source_z)
    if int(finite.sum()) < 3:
        raise ValueError("Pixel grid contains fewer than three finite cells")
    pixel_points = np.column_stack([source_grid_x[finite], source_grid_y[finite]])
    map_points = transform_xy(pixel_points, matrix)
    values = source_z[finite]

    if cell_size is None:
        dx = float(np.median(np.diff(source_x))) if len(source_x) > 1 else 1.0
        dy = float(np.median(np.diff(source_y))) if len(source_y) > 1 else 1.0
        area_scale = abs(float(np.linalg.det(matrix[:, :2])))
        cell_size = math.sqrt(max(area_scale * dx * dy, 1e-9))
    if cell_size <= 0:
        raise ValueError("Cell size must be positive")

    minimum = np.floor(map_points.min(axis=0) / cell_size + 1e-9) * cell_size
    maximum = np.ceil(map_points.max(axis=0) / cell_size - 1e-9) * cell_size
    target_x = np.arange(minimum[0], maximum[0] + cell_size * 0.5, cell_size)
    target_y = np.arange(minimum[1], maximum[1] + cell_size * 0.5, cell_size)
    if len(target_x) * len(target_y) > 4_000_000:
        raise ValueError("Target grid exceeds four million cells; increase cell size")
    grid_x, grid_y = np.meshgrid(target_x, target_y)
    interpolator = LinearNDInterpolator(map_points, values, fill_value=np.nan)
    target_z = np.asarray(interpolator(grid_x, grid_y), dtype=float)
    crs = CRS.from_user_input(target_crs)
    grid = Grid(x=target_x, y=target_y, z=target_z, crs=crs)

    output_dir.mkdir(parents=True, exist_ok=True)
    cps_path = output_dir / f"{name}.cps3"
    xyz_path = output_dir / f"{name}.xyz"
    prj_path = output_dir / f"{name}.prj"
    metadata_path = output_dir / f"{name}.json"
    write_cps3(grid, cps_path, name)
    write_xyz(grid, xyz_path)
    prj_path.write_text(crs.to_wkt("WKT1_ESRI"), encoding="ascii")

    finite_target = target_z[np.isfinite(target_z)]
    independently_checked = len(control_points) >= 4
    accepted = independently_checked and control_quality["p95_m"] <= cell_size * 0.5
    metadata = {
        "status": "accepted" if accepted else "review",
        "source": str(pixel_grid_path),
        "target_crs": crs.to_string(),
        "target_crs_name": crs.name,
        "affine_matrix": matrix.tolist(),
        "control_points": control_points,
        "control_quality": control_quality,
        "independently_checked": independently_checked,
        "cell_size_m": float(cell_size),
        "columns": len(target_x),
        "rows": len(target_y),
        "finite_cells": int(len(finite_target)),
        "bounds": [
            float(target_x.min()),
            float(target_y.min()),
            float(target_x.max()),
            float(target_y.max()),
        ],
        "z_min_m": float(finite_target.min()),
        "z_max_m": float(finite_target.max()),
        "files": {"cps3": str(cps_path), "xyz": str(xyz_path), "prj": str(prj_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Georeference a pixel NPZ grid and export CPS-3")
    parser.add_argument("pixel_grid", type=Path)
    parser.add_argument("control_points", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-crs", required=True)
    parser.add_argument("--cell-size", type=float)
    parser.add_argument("--name", default="digitized_surface")
    args = parser.parse_args()
    controls = json.loads(args.control_points.read_text(encoding="utf-8"))
    result = georeference_grid(
        args.pixel_grid,
        controls,
        args.output_dir,
        target_crs=args.target_crs,
        cell_size=args.cell_size,
        name=args.name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
