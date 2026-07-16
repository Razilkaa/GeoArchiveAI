"""Assign OCR contour labels to traced map lines in pixel coordinates."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_PATTERN = re.compile(r"^-[0-9]+\.[0-9]{1,2}$")


def straightness(points: np.ndarray) -> float:
    chord = points[-1] - points[0]
    length = np.linalg.norm(chord)
    if length < 1:
        return 0.0
    offsets = np.abs(
        chord[0] * (points[:, 1] - points[0, 1])
        - chord[1] * (points[:, 0] - points[0, 0])
    ) / length
    return float(offsets.max() / length)


def normalized_label(text: str, interval: float, tolerance: float) -> float | None:
    text = text.strip().replace(",", ".").replace("−", "-").replace("–", "-")
    if not LABEL_PATTERN.match(text):
        return None
    value = float(text)
    snapped = round(value / interval) * interval
    return round(snapped, 3) if abs(value - snapped) <= tolerance else None


def assign_values(
    polylines: list[np.ndarray],
    readings: list[dict],
    *,
    interval: float,
    snap_tolerance: float = 0.06,
    max_label_distance: float = 70.0,
) -> dict:
    profile_leaks = []
    for points in polylines:
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        profile_leaks.append(straightness(points) < 0.035 and length > 400)

    labels = []
    rejected = 0
    for reading in readings:
        if reading.get("unreadable") or reading.get("zone") != "map_body":
            continue
        quad = np.asarray(reading.get("quad", []), dtype=float)
        if quad.shape != (4, 2):
            continue
        center = quad.mean(axis=0)
        for item in reading.get("values") or []:
            raw_text = str(item.get("text", "")).strip().replace(",", ".")
            raw_text = raw_text.replace("−", "-").replace("–", "-")
            if not LABEL_PATTERN.match(raw_text):
                continue
            value = normalized_label(raw_text, interval, snap_tolerance)
            if value is None:
                rejected += 1
            else:
                labels.append((float(center[0]), float(center[1]), value))

    trees = [cKDTree(points) for points in polylines]
    votes: dict[int, list[float]] = {}
    unmatched = 0
    for x, y, value in labels:
        candidates = [
            (float(tree.query([x, y])[0]), index)
            for index, tree in enumerate(trees)
            if not profile_leaks[index]
        ]
        distance, index = min(candidates, default=(float("inf"), -1))
        if distance <= max_label_distance:
            votes.setdefault(index, []).append(value)
        else:
            unmatched += 1

    values = {}
    confident = []
    conflicts = []
    for index, candidates in votes.items():
        counts = Counter(candidates)
        value, count = counts.most_common(1)[0]
        values[index] = value
        if len(counts) == 1 or count >= 2 * (len(candidates) - count):
            confident.append(index)
        else:
            conflicts.append(index)
    return {
        "values": values,
        "confident": confident,
        "conflicts": conflicts,
        "profile_leaks": profile_leaks,
        "labels": labels,
        "rejected_labels": rejected,
        "unmatched_labels": unmatched,
    }


def run(
    isolines_path: Path,
    readings_path: Path,
    image_path: Path,
    output_dir: Path,
    *,
    interval: float,
) -> dict:
    payload = json.loads(isolines_path.read_text(encoding="utf-8"))
    polylines = [np.asarray(item, dtype=float) for item in payload["polylines_xy"]]
    readings = [
        json.loads(line) for line in readings_path.read_text(encoding="utf-8").splitlines() if line
    ]
    Image.MAX_IMAGE_PIXELS = None
    image = np.asarray(Image.open(image_path).convert("L"))
    ocr_width, ocr_height = image.shape[1], image.shape[0]
    baseline_path = readings_path.with_name("baseline_metrics.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        match = re.search(r"(\d+)x(\d+)\s*px", str(baseline.get("sheet", "")))
        if match:
            ocr_width, ocr_height = int(match.group(1)), int(match.group(2))
    scale_x = image.shape[1] / ocr_width
    scale_y = image.shape[0] / ocr_height
    if abs(scale_x - 1.0) > 1e-4 or abs(scale_y - 1.0) > 1e-4:
        for reading in readings:
            reading["quad"] = [
                [float(x) * scale_x, float(y) * scale_y]
                for x, y in reading.get("quad") or []
            ]
    result = assign_values(polylines, readings, interval=interval)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for index, points in enumerate(polylines):
        is_profile = result["profile_leaks"][index]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": index,
                    "kind": "seismic_profile" if is_profile else "isoline",
                    "value_km": result["values"].get(index),
                    "confident": index in result["confident"],
                    "source": "ocr_label" if index in result["values"] else None,
                },
                "geometry": {"type": "LineString", "coordinates": points.tolist()},
            }
        )
    geojson_path = output_dir / "valued_contours_pixels.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(16, 13), dpi=120)
    ax.imshow(image, cmap="gray", alpha=0.25)
    values = list(result["values"].values())
    low, high = (min(values), max(values)) if values else (-5.0, -1.0)
    cmap = plt.get_cmap("turbo")
    for index, points in enumerate(polylines):
        if result["profile_leaks"][index]:
            color, width, style = "#9aa7b8", 0.7, ":"
        elif index in result["values"]:
            ratio = (result["values"][index] - low) / max(0.1, high - low)
            color = cmap(ratio)
            width = 2.2 if index in result["confident"] else 1.4
            style = "-" if index in result["confident"] else "--"
        else:
            color, width, style = "#999999", 0.8, "-"
        ax.plot(points[:, 0], points[:, 1], color=color, lw=width, ls=style)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    ax.set_title(
        f"Contour assignment: {len(result['values'])}/{len(polylines)} valued; "
        f"{len(result['confident'])} confident; interval {interval:g} km"
    )
    preview_path = output_dir / "valued_contours_preview.png"
    fig.savefig(preview_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "source": str(isolines_path),
        "polylines": len(polylines),
        "profile_leaks": int(sum(result["profile_leaks"])),
        "ocr_labels": len(result["labels"]),
        "rejected_labels": result["rejected_labels"],
        "unmatched_labels": result["unmatched_labels"],
        "valued_polylines": len(result["values"]),
        "confident_polylines": len(result["confident"]),
        "conflicting_polylines": len(result["conflicts"]),
        "direct_value_rate": round(len(result["values"]) / max(1, len(polylines)), 4),
        "contour_interval_km": interval,
        "ocr_image_size": [ocr_width, ocr_height],
        "trace_image_size": [image.shape[1], image.shape[0]],
        "ocr_to_trace_scale": [round(scale_x, 6), round(scale_y, 6)],
        "files": {"geojson": str(geojson_path), "preview": str(preview_path)},
    }
    (output_dir / "value_assignment_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolines", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, required=True)
    args = parser.parse_args()
    result = run(
        args.isolines,
        args.readings,
        args.image,
        args.output_dir,
        interval=args.interval,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
