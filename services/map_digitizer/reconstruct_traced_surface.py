"""Iteratively reconstruct a surface from sparsely labelled traced contours."""
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

from services.map_digitizer.export_cps3_grid import (
    build_harmonic_grid,
    contour_topology,
    extract_surface_contours,
)
from services.map_digitizer.trace_guided_surface import densify, prune_crossing_constraints

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def infer_from_grid(
    support: gpd.GeoDataFrame,
    grid,
    known_ids: set[int],
    *,
    interval: float,
) -> list[dict]:
    interpolator = RegularGridInterpolator(
        (grid.y, grid.x), grid.z / 1000.0, bounds_error=False, fill_value=np.nan
    )
    inferred = []
    for row in support.itertuples():
        trace_id = int(row.trace_id)
        if trace_id in known_ids:
            continue
        samples = densify(np.asarray(row.geometry.coords), max(10.0, interval * 100.0))
        values = interpolator(np.column_stack([samples[:, 1], samples[:, 0]]))
        valid = np.isfinite(values)
        coverage = float(np.mean(valid))
        if np.sum(valid) < 5 or coverage < 0.5:
            continue
        values = values[valid]
        median = float(np.median(values))
        snapped = round(median / interval) * interval
        spread = float(np.percentile(values, 90) - np.percentile(values, 10))
        residual = float(np.median(np.abs(values - snapped)))
        if spread <= interval * 1.5 and residual <= interval * 0.6:
            inferred.append(
                {
                    "id": trace_id,
                    "accepted": True,
                    "value_km": round(snapped, 3),
                    "direct_label": False,
                    "coverage": coverage,
                    "spread_km": spread,
                    "residual_km": residual,
                    "geometry": row.geometry,
                }
            )
    return inferred


def run(
    assigned_path: Path,
    image_path: Path,
    output_dir: Path,
    *,
    interval: float,
    iterations: int = 2,
) -> dict:
    assigned = gpd.read_file(assigned_path)
    assigned["trace_id"] = assigned["id"].astype(int)
    line_features = assigned[assigned["kind"] == "isoline"].copy()
    support = line_features[["trace_id", "geometry"]].copy().set_crs("EPSG:3857", allow_override=True)
    trusted = assigned[
        (assigned["confident"] == True)
        & assigned["value_km"].notna()
        & assigned["kind"].isin(["isoline", "contour_label"])
    ]
    constraints = [
        {
            "id": int(row.trace_id),
            "accepted": True,
            "value_km": float(row.value_km),
            "direct_label": True,
            "coverage": 1.0,
            "spread_km": 0.0,
            "residual_km": 0.0,
            "geometry": row.geometry,
        }
        for row in trusted.itertuples()
    ]
    if len(constraints) < 3:
        raise ValueError(f"Only {len(constraints)} trusted contours; at least 3 required")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as image:
        image_size = image.size
        background = np.asarray(image.convert("L"))
    cell_size = max(12.0, min(image_size) / 280.0)
    blanking = min(image_size) / 10.0
    iteration_metrics = []
    grid = None
    quality = None
    for iteration in range(iterations + 1):
        active = [item for item in constraints if item["accepted"]]
        contour_frame = gpd.GeoDataFrame(
            [{"trace_id": item["id"], "value_km": item["value_km"], "geometry": item["geometry"]} for item in active],
            geometry="geometry",
            crs="EPSG:3857",
        )
        grid, quality = build_harmonic_grid(
            contour_frame,
            cell_size=cell_size,
            blanking_distance=blanking,
            constraint_weight=30.0,
            support_contours=support.assign(value_km=0.0),
        )
        if iteration == iterations:
            break
        known = {item["id"] for item in active}
        candidates = infer_from_grid(support, grid, known, interval=interval)
        before = len(candidates)
        constraints.extend(candidates)
        removed = prune_crossing_constraints(constraints)
        added = sum(item["accepted"] and item["id"] not in known for item in constraints)
        iteration_metrics.append(
            {"iteration": iteration + 1, "candidates": before, "added": added, "crossing_rejections": removed}
        )
        if added == 0:
            break

    final_constraints = [item for item in constraints if item["accepted"]]
    reconstructed = extract_surface_contours(grid, interval=interval * 1000.0)
    topology = contour_topology(reconstructed)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_frame = gpd.GeoDataFrame(
        [
            {
                "trace_id": item["id"], "value_km": item["value_km"],
                "source": "ocr_label" if item["direct_label"] else "surface_inference",
                "geometry": item["geometry"],
            }
            for item in final_constraints
        ],
        geometry="geometry", crs="EPSG:3857",
    )
    source_path = output_dir / "valued_source_contours_pixels.geojson"
    source_frame.to_file(source_path, driver="GeoJSON")
    final_path = output_dir / "reconstructed_contours_pixels.geojson"
    reconstructed.to_file(final_path, driver="GeoJSON")
    grid_path = output_dir / "surface_pixels.npz"
    np.savez_compressed(grid_path, x=grid.x, y=grid.y, z=grid.z)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=120)
    axes[0].imshow(background, cmap="gray")
    direct = source_frame[source_frame["source"] == "ocr_label"]
    inferred = source_frame[source_frame["source"] != "ocr_label"]
    inferred.plot(ax=axes[0], color="#00a36c", linewidth=1.0)
    direct.plot(ax=axes[0], color="#ff365e", linewidth=1.7)
    axes[0].set_title(f"Source contours: {len(direct)} OCR + {len(inferred)} inferred")
    axes[0].axis("off")
    fill = axes[1].pcolormesh(grid.x, grid.y, np.ma.masked_invalid(grid.z), shading="auto", cmap="viridis_r")
    reconstructed.plot(ax=axes[1], color="#111111", linewidth=0.7)
    source_frame.plot(ax=axes[1], color="white", linewidth=0.45, alpha=0.7)
    axes[1].invert_yaxis(); axes[1].set_aspect("equal"); axes[1].axis("off")
    axes[1].set_title("Topology-safe reconstructed surface")
    fig.colorbar(fill, ax=axes[1], shrink=0.75, label="Depth, m")
    preview = output_dir / "surface_preview.png"
    fig.savefig(preview, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "traces": len(support),
        "direct_constraints": len(trusted),
        "direct_line_constraints": int(sum(trusted.geometry.geom_type == "LineString")),
        "direct_point_constraints": int(sum(trusted.geometry.geom_type == "Point")),
        "final_constraints": len(final_constraints),
        "iterations": iteration_metrics,
        "grid_quality": quality,
        "topology": topology,
        "files": {"source_contours": str(source_path), "final_contours": str(final_path), "grid": str(grid_path), "preview": str(preview)},
    }
    (output_dir / "surface_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assigned", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.assigned, args.image, args.output_dir, interval=args.interval, iterations=args.iterations), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
