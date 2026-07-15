from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def accepted_points(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, float]:
    numeric = []
    for item in items:
        reading = item.get("reading", {})
        value = reading.get("parsed_value")
        if reading.get("readable") and reading.get("class") == "numeric" and value is not None:
            numeric.append(float(value))
    median = statistics.median(numeric) if numeric else 0.0
    mad = statistics.median(abs(value - median) for value in numeric) if numeric else 0.0
    robust_limit = max(0.75, 6 * mad)
    rows = []
    for item in items:
        reading = item.get("reading", {})
        value = reading.get("parsed_value")
        flags = list(reading.get("qc_flags") or [])
        normalized_text = str(reading.get("text") or "").strip().replace(",", ".")
        if reading.get("class") == "numeric" and not re.fullmatch(r"-?\d\.\d{2}", normalized_text):
            flags.append("non_measurement_format")
        if item.get("font_band") != "small":
            flags.append("not_small_measurement_font")
        bbox = item.get("bbox") or [0, 0, 0, 0]
        center_x = float(item.get("cx", bbox[0] + bbox[2] / 2))
        center_y = float(item.get("cy", bbox[1] + bbox[3] / 2))
        if value is not None and abs(float(value) - median) > robust_limit:
            flags.append("robust_value_outlier")
        accepted = (
            reading.get("readable")
            and reading.get("class") == "numeric"
            and value is not None
            and not flags
        )
        rows.append(
            {
                "id": int(item["id"]),
                "x_px": center_x,
                "y_px": center_y,
                "value": float(value) if value is not None else None,
                "text": reading.get("text"),
                "class": reading.get("class"),
                "readable": bool(reading.get("readable")),
                "accepted": bool(accepted),
                "qc_flags": flags,
                "bbox": item.get("bbox"),
                "angle": item.get("angle"),
                "detector_confidence": item.get("detector_confidence"),
                "font_band": item.get("font_band"),
            }
        )
    candidates = [row for row in rows if row["accepted"]]
    if len(candidates) >= 10:
        coords = np.array([[row["x_px"], row["y_px"]] for row in candidates])
        values = np.array([row["value"] for row in candidates])
        tree = cKDTree(coords)
        neighbour_count = min(9, len(candidates))
        _, indices = tree.query(coords, k=neighbour_count)
        local_median = np.median(values[indices[:, 1:]], axis=1)
        local_tolerance = max(0.22, mad * 1.5)
        for row, local_value in zip(candidates, local_median):
            if abs(row["value"] - local_value) > local_tolerance:
                row["qc_flags"].append("local_spatial_outlier")
                row["accepted"] = False
    return rows, median, robust_limit


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "id", "x_px", "y_px", "value", "text", "class", "readable", "accepted",
        "qc_flags", "bbox", "angle", "detector_confidence", "font_band",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "qc_flags": ";".join(row["qc_flags"]), "bbox": json.dumps(row["bbox"])})


def write_geojson(rows: list[dict[str, Any]], path: Path) -> None:
    features = []
    for row in rows:
        properties = {key: value for key, value in row.items() if key not in {"x_px", "y_px", "bbox"}}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["x_px"], row["y_px"]]},
                "properties": properties,
            }
        )
    payload = {
        "type": "FeatureCollection",
        "name": "digitized_map_points_pixel_coordinates",
        "pixel_coordinate_note": "Origin is the upper-left image corner; y increases downward. Not georeferenced.",
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render(source: Path, rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as original:
        width, height = original.size
        preview = original.convert("RGB")
        preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    sx, sy = preview.width / width, preview.height / height
    accepted = [row for row in rows if row["accepted"]]
    x = np.array([row["x_px"] * sx for row in accepted])
    y = np.array([row["y_px"] * sy for row in accepted])
    values = np.array([row["value"] for row in accepted])

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.imshow(preview, cmap="gray")
    points = ax.scatter(x, y, c=values, cmap="turbo", s=14, edgecolors="black", linewidths=0.2)
    fig.colorbar(points, ax=ax, fraction=0.025, pad=0.015, label="Digitized value")
    ax.set_title(f"Digitized values: {len(accepted)} accepted points (pixel coordinates)")
    ax.axis("off")
    fig.tight_layout()
    points_path = output_dir / "measured_points_overlay.png"
    fig.savefig(points_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    gx = np.linspace(float(x.min()), float(x.max()), 520)
    gy = np.linspace(float(y.min()), float(y.max()), 520)
    grid_x, grid_y = np.meshgrid(gx, gy)
    rbf = RBFInterpolator(
        np.column_stack([x, y]),
        values,
        kernel="thin_plate_spline",
        smoothing=2.0,
        neighbors=min(60, len(values)),
    )
    grid = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    grid = gaussian_filter(grid, 1.2)
    tree = cKDTree(np.column_stack([x, y]))
    nearest, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
    if len(x) > 1:
        point_spacing = tree.query(np.column_stack([x, y]), k=2)[0][:, 1]
        blanking_distance = max(35.0, float(np.median(point_spacing)) * 5)
    else:
        blanking_distance = 35.0
    grid.reshape(-1)[nearest > blanking_distance] = np.nan
    low = math.floor(float(np.nanpercentile(values, 2)) * 10) / 10
    high = math.ceil(float(np.nanpercentile(values, 98)) * 10) / 10
    levels = np.arange(low, high + 0.05, 0.1)

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.imshow(preview, cmap="gray", alpha=0.48)
    contours = ax.contour(grid_x, grid_y, grid, levels=levels, colors="#b30000", linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.1f")
    ax.scatter(x, y, c=values, cmap="turbo", s=8, alpha=0.75)
    ax.set_title("Point interpolation QC from measured values (not a digitized map)")
    ax.axis("off")
    fig.tight_layout()
    contour_path = output_dir / "point_interpolation_qc.png"
    fig.savefig(contour_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "source_width": width,
        "source_height": height,
        "accepted_points": len(accepted),
        "blanking_distance_preview_px": round(blanking_distance, 2),
        "contour_min": low,
        "contour_max": high,
        "contour_interval": 0.1,
        "measured_points_overlay": str(points_path),
        "point_interpolation_qc": str(contour_path),
    }


def run(results_path: Path, source: Path, output_dir: Path | None = None) -> Path:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    output_dir = output_dir or results_path.parent / "digitized"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, median, robust_limit = accepted_points(payload["items"])
    write_csv(rows, output_dir / "measured_points.csv")
    write_geojson(rows, output_dir / "measured_points.geojson")
    render_metrics = render(source, rows, output_dir)
    summary = {
        "source": str(source),
        "coordinate_system": "image_pixels",
        "georeferenced": False,
        "total_candidates": len(rows),
        "accepted_points": sum(row["accepted"] for row in rows),
        "rejected_points": sum(not row["accepted"] for row in rows),
        "value_median": round(median, 4),
        "robust_limit": round(robust_limit, 4),
        **render_metrics,
    }
    output = output_dir / "digitization_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render digitized map points and reconstructed contours")
    parser.add_argument("results", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(run(args.results, args.source, args.out))


if __name__ == "__main__":
    main()
