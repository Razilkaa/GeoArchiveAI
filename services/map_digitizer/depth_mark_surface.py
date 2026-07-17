"""Reconstruct a pixel-coordinate surface from dense profile depth marks."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString
from skimage.measure import find_contours

from services.map_digitizer.assign_contour_values import dominant_value_band

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEPTH_PATTERN = re.compile(r"^-?[0-9]+\.[0-9]{2}$")
CONTOUR_PATTERN = re.compile(r"^-[0-9]+(?:\.[0-9]{1,3})?$")


def image_scale(readings_path: Path, image_size: tuple[int, int]) -> tuple[float, float, int, int]:
    width, height = image_size
    ocr_width, ocr_height = width, height
    baseline_path = readings_path.with_name("baseline_metrics.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        match = re.search(r"(\d+)x(\d+)\s*px", str(baseline.get("sheet", "")))
        if match:
            ocr_width, ocr_height = int(match.group(1)), int(match.group(2))
    return width / ocr_width, height / ocr_height, ocr_width, ocr_height


def load_ocr_readings(path: Path, image_size: tuple[int, int]) -> list[dict]:
    """Load either crop/VLM JSONL or the PaddleOCR HTTP response."""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        width, height = image_size
        readings = []
        for line in payload["lines"]:
            polygon = line.get("polygon") or []
            quad = np.asarray(polygon, dtype=float)
            if quad.shape != (4, 2):
                continue
            center_x, center_y = quad.mean(axis=0)
            in_map = (
                0.02 * width <= center_x <= 0.98 * width
                and 0.02 * height <= center_y <= 0.82 * height
            )
            readings.append(
                {
                    "zone": "map_body" if in_map else "outside",
                    "quad": quad.tolist(),
                    "values": [{"text": line.get("text", "")}],
                    "unreadable": float(line.get("score") or 0.0) < 0.80,
                    "ocr_confidence": line.get("score"),
                }
            )
        return readings
    return [json.loads(line) for line in text.splitlines() if line]


def extract_measurements(
    readings: list[dict],
    *,
    scale_x: float,
    scale_y: float,
    contour_interval: float,
    value_range: tuple[float, float] = (1.0, 8.0),
) -> tuple[np.ndarray, np.ndarray]:
    marks = []
    labels = []
    for reading in readings:
        if reading.get("unreadable") or reading.get("zone") != "map_body":
            continue
        quad = np.asarray(reading.get("quad") or [], dtype=float)
        if quad.shape != (4, 2):
            continue
        x, y = quad.mean(axis=0) * np.asarray([scale_x, scale_y])
        for item in reading.get("values") or []:
            text = str(item.get("text", "")).strip().replace(",", ".")
            text = text.replace("−", "-").replace("–", "-")
            if DEPTH_PATTERN.match(text):
                value = abs(float(text))
                if value_range[0] <= value <= value_range[1]:
                    marks.append((x, y, value))
            elif CONTOUR_PATTERN.match(text):
                value = abs(float(text))
                if value >= 100.0 and contour_interval < 10.0:
                    value /= 1000.0
                snapped = round(value / contour_interval) * contour_interval
                if abs(value - snapped) <= 0.03:
                    labels.append((x, y, snapped))
    return np.asarray(marks, dtype=float), np.asarray(labels, dtype=float)


def filter_depth_marks(marks: np.ndarray, threshold: float = 0.25, neighbours: int = 9) -> np.ndarray:
    if len(marks) < neighbours:
        return marks
    tree = cKDTree(marks[:, :2])
    _, indices = tree.query(marks[:, :2], k=neighbours)
    local_median = np.median(marks[indices[:, 1:], 2], axis=1)
    return marks[np.abs(marks[:, 2] - local_median) <= threshold]


def fuse_measurement_sources(
    sources: list[tuple[str, np.ndarray]],
    *,
    radius: float,
) -> tuple[np.ndarray, dict]:
    """Fuse precise OCR anchors with strictly filtered high-recall VLM marks."""
    if not sources:
        return np.empty((0, 3)), {"mode": "empty"}
    paddle = [marks for kind, marks in sources if kind == "paddle" and len(marks)]
    vlm = [marks for kind, marks in sources if kind != "paddle" and len(marks)]
    if not paddle or not vlm:
        combined = np.vstack([marks for _, marks in sources if len(marks)])
        filtered = filter_depth_marks(combined)
        return merge_nearby_measurements(filtered, radius), {
            "mode": "single_source",
            "retained": int(len(filtered)),
        }

    precise = filter_depth_marks(np.vstack(paddle), threshold=0.18)
    recall = filter_depth_marks(np.vstack(vlm), threshold=0.08)
    distance, nearest = cKDTree(precise[:, :2]).query(recall[:, :2])
    close = distance <= max(40.0, radius * 1.75)
    agreement = np.abs(recall[:, 2] - precise[nearest, 2]) <= 0.08
    # Precise OCR owns overlapping locations. VLM contributes only new territory.
    supplemental = recall[~close]
    fused = merge_nearby_measurements(np.vstack([precise, supplemental]), radius)
    return fused, {
        "mode": "paddle_anchor_vlm_fill",
        "paddle_retained": int(len(precise)),
        "vlm_locally_consistent": int(len(recall)),
        "overlapping_pairs": int(np.sum(close)),
        "agreeing_pairs": int(np.sum(close & agreement)),
        "agreement_rate": round(float(np.mean(agreement[close])), 4) if np.any(close) else None,
        "vlm_supplemental": int(len(supplemental)),
        "fused_marks": int(len(fused)),
    }


def merge_nearby_measurements(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) < 2:
        return points
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    tree = cKDTree(points[:, :2])
    for left, right in tree.query_pairs(radius):
        union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    merged = []
    for indices in groups.values():
        cluster = points[indices]
        merged.append(
            (
                float(np.median(cluster[:, 0])),
                float(np.median(cluster[:, 1])),
                float(np.median(cluster[:, 2])),
            )
        )
    return np.asarray(merged, dtype=float)


def build_surface(
    marks: np.ndarray,
    image_size: tuple[int, int],
    *,
    columns: int = 500,
    blanking_distance: float = 400.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RBFInterpolator]:
    width, height = image_size
    rows = max(2, round(columns * height / width))
    x = np.linspace(0.0, width - 1.0, columns)
    y = np.linspace(0.0, height - 1.0, rows)
    grid_x, grid_y = np.meshgrid(x, y)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    interpolator = RBFInterpolator(
        marks[:, :2],
        marks[:, 2],
        kernel="thin_plate_spline",
        smoothing=10000.0,
        neighbors=min(50, len(marks)),
    )
    values = interpolator(nodes)
    distance, _ = cKDTree(marks[:, :2]).query(nodes)
    inside = Delaunay(marks[:, :2]).find_simplex(nodes) >= 0
    values[(distance > blanking_distance) | ~inside] = np.nan
    return x, y, values.reshape(grid_x.shape), interpolator


def vectorize_surface(
    x: np.ndarray,
    y: np.ndarray,
    surface: np.ndarray,
    interval: float,
) -> tuple[list[dict], dict]:
    finite = surface[np.isfinite(surface)]
    first = math.ceil(float(finite.min()) / interval) * interval
    last = math.floor(float(finite.max()) / interval) * interval
    mask = np.isfinite(surface)
    features = []
    lines = []
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    for level in np.arange(first, last + interval * 0.5, interval):
        for path in find_contours(surface, float(level), mask=mask):
            if len(path) < 5:
                continue
            coordinates = [
                [float(x[0] + point[1] * dx), float(y[0] + point[0] * dy)]
                for point in path
            ]
            line = LineString(coordinates)
            if line.length < max(dx, dy) * 3:
                continue
            lines.append(line)
            features.append(
                {
                    "type": "Feature",
                    "properties": {"value_km": -float(level)},
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                }
            )
    crossings = sum(
        lines[left].crosses(lines[right])
        for left in range(len(lines))
        for right in range(left + 1, len(lines))
    )
    return features, {"segments": len(lines), "crossing_pairs": int(crossings)}


def run(
    readings_path: Path | list[Path],
    image_path: Path,
    output_dir: Path,
    *,
    interval: float,
) -> dict:
    Image.MAX_IMAGE_PIXELS = None
    image = np.asarray(Image.open(image_path).convert("L"))
    paths = [readings_path] if isinstance(readings_path, Path) else list(readings_path)
    mark_parts: list[tuple[str, np.ndarray]] = []
    label_parts: list[tuple[str, np.ndarray]] = []
    source_metrics = []
    for path in paths:
        scale_x, scale_y, ocr_width, ocr_height = image_scale(
            path, (image.shape[1], image.shape[0])
        )
        readings = load_ocr_readings(path, (ocr_width, ocr_height))
        source_marks, source_labels = extract_measurements(
            readings,
            scale_x=scale_x,
            scale_y=scale_y,
            contour_interval=interval,
        )
        source_kind = "paddle" if path.suffix.lower() == ".json" else "vlm"
        mark_parts.append((source_kind, source_marks))
        label_parts.append((source_kind, source_labels))
        source_metrics.append(
            {
                "path": str(path),
                "kind": source_kind,
                "depth_marks": int(len(source_marks)),
                "contour_labels": int(len(source_labels)),
                "ocr_image_size": [ocr_width, ocr_height],
                "ocr_to_trace_scale": [round(scale_x, 6), round(scale_y, 6)],
            }
        )
    available_marks = [part for _, part in mark_parts if len(part)]
    raw_marks = np.vstack(available_marks) if available_marks else np.empty((0, 3))
    labels = np.vstack([part for _, part in label_parts if len(part)]) if any(len(part) for _, part in label_parts) else np.empty((0, 3))
    dedupe_radius = max(8.0, min(image.shape) * 0.003)
    raw_count_before_merge = len(raw_marks)
    labels = merge_nearby_measurements(labels, dedupe_radius)
    raw_label_count = len(labels)
    label_band = dominant_value_band(labels[:, 2].tolist(), interval) if len(labels) else None
    if label_band is not None:
        labels = labels[(labels[:, 2] >= label_band[0]) & (labels[:, 2] <= label_band[1])]
    marks, fusion_metrics = fuse_measurement_sources(mark_parts, radius=dedupe_radius)
    surface_mode = "dense_profile_measurements"
    if len(marks) < 20 and len(labels) >= 5:
        marks = labels.copy()
        surface_mode = "sparse_contour_labels"
    if len(marks) < 5:
        raise ValueError(f"Only {len(marks)} valid depth constraints; at least 5 are required")
    x, y, surface, interpolator = build_surface(
        marks, (image.shape[1], image.shape[0])
    )

    def crosscheck(check_labels: np.ndarray) -> dict:
        result = {"candidates": int(len(check_labels)), "compared": 0}
        if not len(check_labels):
            return result
        nearest, _ = cKDTree(marks[:, :2]).query(check_labels[:, :2])
        expected_low = max(0.0, float(np.percentile(marks[:, 2], 1) - 0.6))
        expected_high = float(np.percentile(marks[:, 2], 99) + 0.6)
        eligible = (
            (nearest <= 250.0)
            & (check_labels[:, 2] >= expected_low)
            & (check_labels[:, 2] <= expected_high)
        )
        checked = check_labels[eligible]
        if len(checked):
            error = np.abs(interpolator(checked[:, :2]) - checked[:, 2])
            result.update(
                {
                    "compared": int(len(error)),
                    "median_abs_error_km": float(np.median(error)),
                    "p90_abs_error_km": float(np.percentile(error, 90)),
                    "within_one_interval_rate": float(np.mean(error <= interval)),
                }
            )
        return result

    label_metrics = crosscheck(labels)
    label_metrics["by_source"] = {
        kind: crosscheck(merge_nearby_measurements(np.vstack(parts), dedupe_radius))
        for kind in {kind for kind, _ in label_parts}
        if (parts := [part for source_kind, part in label_parts if source_kind == kind and len(part)])
    }

    features, topology = vectorize_surface(x, y, surface, interval)
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = output_dir / "depth_mark_surface_isolines_pixels.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    npz_path = output_dir / "depth_mark_surface_pixels.npz"
    np.savez_compressed(npz_path, x=x, y=y, z=-surface)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=120)
    preview_step = max(1, int(np.ceil(max(image.shape) / 3000.0)))
    preview_image = image[::preview_step, ::preview_step]
    preview_extent = (0, image.shape[1], image.shape[0], 0)
    axes[0].imshow(preview_image, cmap="gray", extent=preview_extent)
    axes[0].set_title("Source map")
    axes[0].axis("off")
    axes[1].imshow(preview_image, cmap="gray", alpha=0.18, extent=preview_extent)
    levels = np.arange(
        math.ceil(float(np.nanmin(surface)) / interval) * interval,
        float(np.nanmax(surface)) + interval * 0.5,
        interval,
    )
    filled = axes[1].contourf(x, y, surface, levels=levels, cmap="viridis_r", alpha=0.8)
    lines = axes[1].contour(x, y, surface, levels=levels, colors="#111111", linewidths=0.55)
    axes[1].clabel(lines, fmt=lambda value: f"-{value:.1f}", fontsize=6)
    axes[1].scatter(marks[:, 0], marks[:, 1], s=2, color="white", alpha=0.65)
    axes[1].set_xlim(0, image.shape[1])
    axes[1].set_ylim(image.shape[0], 0)
    axes[1].set_title(f"Surface from {len(marks)} filtered depth constraints")
    axes[1].axis("off")
    fig.colorbar(filled, ax=axes[1], shrink=0.7, label="Depth magnitude, km")
    preview_path = output_dir / "depth_mark_surface_preview.png"
    fig.savefig(preview_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "surface_mode": surface_mode,
        "raw_depth_marks": int(len(raw_marks)),
        "raw_depth_marks_before_merge": int(raw_count_before_merge),
        "filtered_depth_marks": int(len(marks)),
        "retained_mark_rate": round(
            len(marks)
            / max(1, len(raw_marks) if surface_mode == "dense_profile_measurements" else raw_label_count),
            4,
        ),
        "fusion": fusion_metrics,
        "trace_image_size": [image.shape[1], image.shape[0]],
        "ocr_sources": source_metrics,
        "dedupe_radius_px": round(dedupe_radius, 2),
        "contour_interval_km": interval,
        "raw_contour_labels": int(raw_label_count),
        "retained_contour_labels": int(len(labels)),
        "contour_label_band_km": list(label_band) if label_band else None,
        "label_crosscheck": label_metrics,
        "topology": topology,
        "files": {
            "surface": str(npz_path),
            "isolines": str(geojson_path),
            "preview": str(preview_path),
        },
    }
    (output_dir / "depth_mark_surface_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readings", type=Path, nargs="+", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, required=True)
    args = parser.parse_args()
    result = run(args.readings, args.image, args.output_dir, interval=args.interval)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
