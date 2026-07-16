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


LABEL_PATTERN = re.compile(r"^-[0-9]+(?:\.[0-9]{1,2})?$")


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
    # Soviet maps commonly label the same depth either as -2.8 km or -2800 m.
    if abs(value) >= 100.0 and interval < 10.0:
        value /= 1000.0
    snapped = round(value / interval) * interval
    return round(snapped, 3) if abs(value - snapped) <= tolerance else None


def dominant_value_band(values: list[float], interval: float) -> tuple[float, float] | None:
    """Keep the repeated contour band and reject distant profile-number clusters."""
    if len(values) < 6:
        return None
    ordered = np.sort(np.asarray(values, dtype=float))
    gaps = np.diff(ordered)
    split_points = np.where(gaps > max(interval * 4.0, 0.8))[0]
    groups = np.split(ordered, split_points + 1)
    group = max(groups, key=lambda item: (len(item), -np.ptp(item)))
    if len(group) < max(3, len(values) * 0.25):
        return None
    return float(group.min() - interval), float(group.max() + interval)


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
    profile_labels = []
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
            integer_match = re.fullmatch(r"-?([0-9]{4,6})", raw_text)
            if integer_match:
                integer = int(integer_match.group(1))
                metre_step = max(1, round(interval * 1000.0))
                if integer % metre_step != 0:
                    profile_labels.append((float(center[0]), float(center[1]), integer))
            if not LABEL_PATTERN.match(raw_text):
                continue
            value = normalized_label(raw_text, interval, snap_tolerance)
            if value is None:
                rejected += 1
            else:
                labels.append((float(center[0]), float(center[1]), value))

    for x, y, _ in profile_labels:
        candidates = []
        for index, points in enumerate(polylines):
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length < 140 or straightness(points) >= 0.08:
                continue
            candidates.append((float(cKDTree(points).query([x, y])[0]), index))
        distance, index = min(candidates, default=(float("inf"), -1))
        if distance <= max_label_distance * 1.25:
            profile_leaks[index] = True

    band = dominant_value_band([item[2] for item in labels], interval)
    if band is not None:
        labels = [item for item in labels if band[0] <= item[2] <= band[1]]

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
        "profile_labels": profile_labels,
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
    raw_readings = readings_path.read_text(encoding="utf-8")
    try:
        paddle_payload = json.loads(raw_readings)
    except json.JSONDecodeError:
        paddle_payload = None
    if isinstance(paddle_payload, dict) and isinstance(paddle_payload.get("lines"), list):
        readings = []
        for line in paddle_payload["lines"]:
            readings.append(
                {
                    "zone": "map_body",
                    "quad": line.get("polygon"),
                    "values": [{"text": line.get("text", "")}],
                    "unreadable": float(line.get("score") or 0.0) < 0.75,
                }
            )
    else:
        readings = [json.loads(line) for line in raw_readings.splitlines() if line]
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
    max_label_distance = max(70.0, min(image.shape) * 0.02)
    result = assign_values(
        polylines, readings, interval=interval, max_label_distance=max_label_distance
    )
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
    for label_index, (x, y, value) in enumerate(result["labels"]):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": len(polylines) + label_index,
                    "kind": "contour_label",
                    "value_km": value,
                    "confident": True,
                    "source": "ocr_label_point",
                },
                "geometry": {"type": "Point", "coordinates": [x, y]},
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
        "point_constraints": len(result["labels"]),
        "profile_id_labels": len(result["profile_labels"]),
        "rejected_labels": result["rejected_labels"],
        "unmatched_labels": result["unmatched_labels"],
        "valued_polylines": len(result["values"]),
        "confident_polylines": len(result["confident"]),
        "conflicting_polylines": len(result["conflicts"]),
        "direct_value_rate": round(len(result["values"]) / max(1, len(polylines)), 4),
        "contour_interval_km": interval,
        "max_label_distance_px": round(max_label_distance, 2),
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
