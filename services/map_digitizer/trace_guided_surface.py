"""Snap traced source-map contours to an OCR-derived depth surface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString

from services.map_digitizer.assign_contour_values import assign_values, straightness
from services.map_digitizer.contour_cleanup import remove_unsupported_closed_contours
from services.map_digitizer.depth_mark_surface import image_scale, load_ocr_readings
from services.map_digitizer.export_cps3_grid import (
    build_harmonic_grid,
    contour_topology,
    extract_surface_contours,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    line = LineString(points)
    count = max(2, math.ceil(line.length / spacing) + 1)
    return np.asarray(
        [[point.x, point.y] for point in (line.interpolate(d) for d in np.linspace(0, line.length, count))]
    )


def assign_traced_values(
    polylines: list[np.ndarray],
    interpolator: RegularGridInterpolator,
    *,
    interval: float,
    sample_spacing: float = 20.0,
    minimum_coverage: float = 0.2,
    maximum_spread: float = 0.4,
    maximum_residual: float = 0.12,
    minimum_length: float = 0.0,
) -> list[dict]:
    assignments = []
    for index, points in enumerate(polylines):
        samples = densify(points, sample_spacing)
        predictions = interpolator(np.column_stack([samples[:, 1], samples[:, 0]]))
        valid = np.isfinite(predictions)
        coverage = float(np.mean(valid))
        length = float(LineString(points).length)
        curvature = straightness(points)
        profile_suspect = (
            (curvature < 0.01 and length > 100.0)
            or (curvature < 0.035 and length > 400.0)
        )
        if np.sum(valid) >= 5:
            values = np.abs(predictions[valid])
            median = float(np.median(values))
            snapped = round(median / interval) * interval
            spread = float(np.percentile(values, 90) - np.percentile(values, 10))
            residual = float(np.median(np.abs(values - snapped)))
        else:
            snapped = spread = residual = None
        accepted = bool(
            not profile_suspect
            and length >= minimum_length
            and coverage >= minimum_coverage
            and residual is not None
            and residual <= maximum_residual
            and spread <= maximum_spread
        )
        assignments.append(
            {
                "id": index,
                "accepted": accepted,
                "profile_suspect": profile_suspect,
                "coverage": round(coverage, 4),
                "length_px": round(length, 2),
                "value_km": -round(float(snapped), 3) if accepted else None,
                "spread_km": round(float(spread), 4) if spread is not None else None,
                "residual_km": round(float(residual), 4) if residual is not None else None,
                "geometry": LineString(points),
            }
        )
    return assignments


def prune_crossing_constraints(assignments: list[dict]) -> list[int]:
    """Reject the weaker member of every different-level crossing pair."""
    removed: list[int] = []
    while True:
        accepted = [item for item in assignments if item["accepted"]]
        conflict = None
        for left_index, left in enumerate(accepted):
            for right in accepted[left_index + 1 :]:
                if left["value_km"] != right["value_km"] and left["geometry"].crosses(right["geometry"]):
                    conflict = (left, right)
                    break
            if conflict:
                break
        if not conflict:
            return removed
        left, right = conflict
        def weakness(item: dict) -> tuple[float, float, float]:
            return (
                0.0 if item.get("direct_label") else 1.0,
                float(item.get("residual_km") or 1.0) + float(item.get("spread_km") or 1.0),
                -float(item.get("coverage") or 0.0),
            )
        loser = max((left, right), key=weakness)
        loser["accepted"] = False
        loser["rejection_reason"] = "crosses_different_level"
        removed.append(int(loser["id"]))


def run(
    traces_path: Path,
    point_surface_path: Path,
    output_dir: Path,
    *,
    interval: float,
    trace_image_path: Path | None = None,
    label_paths: list[Path] | None = None,
) -> dict:
    payload = json.loads(traces_path.read_text(encoding="utf-8"))
    raw_polylines = [np.asarray(points, dtype=float) for points in payload["polylines_xy"]]
    surface = np.load(point_surface_path)
    x, y, z = surface["x"], surface["y"], surface["z"]
    target_size = (float(x[-1] + 1), float(y[-1] + 1))

    source_path = trace_image_path or Path(payload.get("source", ""))
    with Image.open(source_path) as trace_image:
        trace_size = trace_image.size
    scale = np.asarray([target_size[0] / trace_size[0], target_size[1] / trace_size[1]])
    polylines = [points * scale for points in raw_polylines]
    interpolator = RegularGridInterpolator(
        (y, x), np.abs(z), bounds_error=False, fill_value=np.nan
    )
    minimum_trace_length = max(60.0, min(target_size) * 0.015)
    assignments = assign_traced_values(
        polylines,
        interpolator,
        interval=interval,
        minimum_length=minimum_trace_length,
    )
    finite_surface = np.abs(z[np.isfinite(z)])
    direct_low = -float(finite_surface.max()) - interval * 2
    direct_high = -float(finite_surface.min()) + interval * 2
    direct_labels: dict[int, float] = {}
    for label_path in label_paths or []:
        scale_x, scale_y, ocr_width, ocr_height = image_scale(
            label_path, (round(target_size[0]), round(target_size[1]))
        )
        readings = load_ocr_readings(label_path, (ocr_width, ocr_height))
        if abs(scale_x - 1.0) > 1e-4 or abs(scale_y - 1.0) > 1e-4:
            for reading in readings:
                reading["quad"] = [
                    [float(px) * scale_x, float(py) * scale_y]
                    for px, py in reading.get("quad") or []
                ]
        direct = assign_values(
            polylines, readings, interval=interval, max_label_distance=110.0
        )
        for index in direct["confident"]:
            value = float(direct["values"][index])
            if direct_low <= value <= direct_high:
                direct_labels[index] = value
    for item in assignments:
        if item["id"] in direct_labels and not item["profile_suspect"]:
            item["value_km"] = direct_labels[item["id"]]
            item["accepted"] = True
            item["direct_label"] = True
        else:
            item["direct_label"] = False
    crossing_rejections = prune_crossing_constraints(assignments)
    accepted = [item for item in assignments if item["accepted"]]
    if len(accepted) < 3:
        raise ValueError(f"Only {len(accepted)} traced contours passed QC")

    source_rows = [
        {
            "trace_id": item["id"],
            "value_km": item["value_km"],
            "coverage": item["coverage"],
            "spread_km": item["spread_km"],
            "residual_km": item["residual_km"],
            "direct_label": item["direct_label"],
            "geometry": item["geometry"],
        }
        for item in accepted
    ]
    contours = gpd.GeoDataFrame(source_rows, geometry="geometry", crs="EPSG:3857")
    support = gpd.GeoDataFrame(
        [
            {"value_km": 0.0, "geometry": item["geometry"]}
            for item in assignments
            if not item["profile_suspect"]
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    grid, grid_quality = build_harmonic_grid(
        contours,
        cell_size=max(15.0, min(target_size) / 250.0),
        blanking_distance=min(target_size) / 12.0,
        constraint_weight=25.0,
        support_contours=support,
    )
    reconstructed = extract_surface_contours(grid, interval=interval * 1000.0)
    grid_spacing = float(np.median(np.diff(grid.x))) if len(grid.x) > 1 else 1.0
    reconstructed, contour_cleanup = remove_unsupported_closed_contours(
        reconstructed,
        contours,
        interval_m=interval * 1000.0,
        support_distance=grid_spacing * 4.0,
    )
    topology = contour_topology(reconstructed)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_geojson = output_dir / "source_aligned_contours_pixels.geojson"
    contours.to_file(source_geojson, driver="GeoJSON")
    final_geojson = output_dir / "reconstructed_contours_pixels.geojson"
    reconstructed.to_file(final_geojson, driver="GeoJSON")
    grid_path = output_dir / "trace_guided_surface_pixels.npz"
    np.savez_compressed(grid_path, x=grid.x, y=grid.y, z=grid.z)

    with Image.open(source_path) as source:
        background = np.asarray(source.convert("L"))
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=120)
    preview_step = max(1, int(np.ceil(max(background.shape) / 3000.0)))
    axes[0].imshow(
        background[::preview_step, ::preview_step],
        cmap="gray",
        extent=(0, background.shape[1], background.shape[0], 0),
    )
    for item in assignments:
        points = np.asarray(item["geometry"].coords) / scale
        color = "#00a36c" if item["accepted"] else "#cc3344" if item["profile_suspect"] else "#999999"
        axes[0].plot(points[:, 0], points[:, 1], color=color, lw=1.0, alpha=0.9)
    axes[0].set_title(f"Source traces: {len(accepted)} accepted")
    axes[0].axis("off")
    masked = np.ma.masked_invalid(grid.z)
    fill = axes[1].pcolormesh(grid.x, grid.y, masked, shading="auto", cmap="viridis_r")
    reconstructed.plot(ax=axes[1], color="#111111", linewidth=0.65)
    contours.plot(ax=axes[1], color="white", linewidth=0.65, alpha=0.9)
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    axes[1].set_title("Trace-guided harmonic surface")
    axes[1].axis("off")
    fig.colorbar(fill, ax=axes[1], shrink=0.75, label="Depth, m")
    preview = output_dir / "trace_guided_surface_preview.png"
    fig.savefig(preview, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "traces": len(assignments),
        "accepted_traces": len(accepted),
        "accepted_rate": round(len(accepted) / max(1, len(assignments)), 4),
        "profile_suspects": sum(item["profile_suspect"] for item in assignments),
        "minimum_trace_length_px": round(minimum_trace_length, 2),
        "direct_labels": len(direct_labels),
        "crossing_rejections": crossing_rejections,
        "trace_image_size": list(trace_size),
        "surface_image_size": [round(target_size[0]), round(target_size[1])],
        "trace_to_surface_scale": [round(float(value), 6) for value in scale],
        "value_distribution": {
            str(value): sum(item["value_km"] == value for item in accepted)
            for value in sorted({item["value_km"] for item in accepted})
        },
        "grid_quality": grid_quality,
        "topology": topology,
        "contour_cleanup": contour_cleanup,
        "files": {
            "source_contours": str(source_geojson),
            "final_contours": str(final_geojson),
            "grid": str(grid_path),
            "preview": str(preview),
        },
    }
    (output_dir / "trace_guided_surface_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--point-surface", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--trace-image", type=Path)
    parser.add_argument("--labels", type=Path, nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(args.traces, args.point_surface, args.output_dir, interval=args.interval, trace_image_path=args.trace_image, label_paths=args.labels), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
