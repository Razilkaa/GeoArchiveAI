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


def fit_similarity(control_points: list[dict]) -> tuple[np.ndarray, dict]:
    """Fit an orientation-reversing similarity from two or more GCPs.

    Scan coordinates grow downwards while projected northings grow upwards,
    hence the reflection. Two distinct point pairs determine translation,
    rotation and one uniform scale; additional points only validate/refine it.
    """
    if len(control_points) < 2:
        raise ValueError("At least two pixel-to-map control points are required")
    pixels = np.asarray([point["pixel"] for point in control_points], dtype=float)
    mapped = np.asarray([point["map"] for point in control_points], dtype=float)
    if pixels.shape != (len(control_points), 2) or mapped.shape != pixels.shape:
        raise ValueError("Each control point must contain pixel [x,y] and map [x,y]")
    if np.linalg.norm(pixels[1] - pixels[0]) < 1.0:
        raise ValueError("Pixel control points must be distinct")
    if np.linalg.norm(mapped[1] - mapped[0]) < 1.0:
        raise ValueError("Map control points must be distinct")
    rows = []
    targets = []
    for (x, y), (map_x, map_y) in zip(pixels, mapped):
        rows.extend(([x, y, 1.0, 0.0], [-y, x, 0.0, 1.0]))
        targets.extend((map_x, map_y))
    parameters, _, rank, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=float),
        np.asarray(targets, dtype=float),
        rcond=None,
    )
    if rank < 4:
        raise ValueError("Control points do not determine a similarity transform")
    a, b, offset_x, offset_y = parameters
    scale = float(math.hypot(a, b))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid control-point scale")
    matrix = np.asarray(
        [[a, b, offset_x], [b, -a, offset_y]],
        dtype=float,
    )
    predicted = transform_xy(pixels, matrix)
    residuals = np.linalg.norm(predicted - mapped, axis=1)
    return matrix, {
        "model": "two_point_similarity",
        "count": len(control_points),
        "scale_m_per_px": scale,
        "rotation_deg": float(math.degrees(math.atan2(b, a))),
        "orientation_reversing": True,
        "pixel_baseline": float(np.linalg.norm(pixels[1] - pixels[0])),
        "map_baseline_m": float(np.linalg.norm(mapped[1] - mapped[0])),
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
    matrix, control_quality = fit_similarity(control_points)
    return _export_transformed_grid(
        pixel_grid_path,
        matrix,
        output_dir,
        target_crs=target_crs,
        cell_size=cell_size,
        name=name,
        transform_metadata={
            "transform_model": "two_point_similarity",
            "control_points": control_points,
            "control_quality": control_quality,
            "independently_checked": len(control_points) >= 3,
        },
    )


def georeference_grid_affine(
    pixel_grid_path: Path,
    affine_matrix: np.ndarray,
    output_dir: Path,
    *,
    target_crs: str,
    cell_size: float | None = None,
    name: str = "digitized_surface",
    validation: dict | None = None,
) -> dict:
    """Resample a pixel grid with an independently validated affine transform."""
    matrix = np.asarray(affine_matrix, dtype=float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Affine matrix must be a finite 2x3 array")
    return _export_transformed_grid(
        pixel_grid_path,
        matrix,
        output_dir,
        target_crs=target_crs,
        cell_size=cell_size,
        name=name,
        transform_metadata={
            "transform_model": "validated_affine",
            "validation": validation or {},
            "independently_checked": True,
        },
    )


def _export_transformed_grid(
    pixel_grid_path: Path,
    matrix: np.ndarray,
    output_dir: Path,
    *,
    target_crs: str,
    cell_size: float | None,
    name: str,
    transform_metadata: dict,
) -> dict:
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
    control_quality = transform_metadata.get("control_quality") or {}
    independently_checked = bool(transform_metadata.get("independently_checked"))
    accepted = independently_checked and (
        not control_quality
        or float(control_quality.get("p95_m", 0.0)) <= max(1.0, cell_size * 0.5)
    )
    metadata = {
        "status": "accepted" if accepted else "review",
        "source": str(pixel_grid_path),
        "target_crs": crs.to_string(),
        "target_crs_name": crs.name,
        **transform_metadata,
        "affine_matrix": matrix.tolist(),
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
