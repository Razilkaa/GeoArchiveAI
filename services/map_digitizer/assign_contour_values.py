"""Assign OCR contour labels to traced map lines in pixel coordinates."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_PATTERN = re.compile(r"^-[0-9]+(?:\.[0-9]{1,3})?$")
INTERVAL_CANDIDATES_KM = (0.5, 0.25, 0.2, 0.1, 0.05, 0.025, 0.02, 0.01)


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


def infer_contour_interval(readings: list[dict]) -> tuple[float, dict]:
    """Infer the largest standard contour step supported by OCR labels."""
    values = []
    for reading in readings:
        if reading.get("unreadable") or reading.get("profile_corridor"):
            continue
        for item in reading.get("values") or []:
            text = str(item.get("text", "")).strip().replace(",", ".")
            text = text.replace("в€’", "-").replace("вЂ“", "-")
            if not LABEL_PATTERN.match(text):
                continue
            value = float(text)
            values.append(value / 1000.0 if abs(value) >= 100.0 else value)
    unique = np.unique(np.round(values, 3))
    if len(unique) < 3:
        raise ValueError("Cannot infer contour interval from fewer than three labels")

    scores = []
    for candidate in INTERVAL_CANDIDATES_KM:
        tolerance = min(0.012, candidate * 0.15)
        support = max(
            float(
                np.mean(
                    np.abs(
                        (unique - anchor)
                        - np.round((unique - anchor) / candidate) * candidate
                    )
                    <= tolerance
                )
            )
            for anchor in unique
        )
        scores.append({"interval_km": candidate, "support": round(support, 4)})
    minimum_support = 0.7
    selected = next(
        (item["interval_km"] for item in scores if item["support"] >= minimum_support),
        max(scores, key=lambda item: item["support"])["interval_km"],
    )
    selected_support = next(
        item["support"] for item in scores if item["interval_km"] == selected
    )
    return float(selected), {
        "label_count": len(values),
        "unique_labels": len(unique),
        "minimum_support": minimum_support,
        "selected_support": selected_support,
        "scores": scores,
    }


def filter_profile_measurements(
    measurements: list[tuple[float, float, float]], interval: float
) -> list[tuple[float, float, float]]:
    if len(measurements) < 7:
        return measurements
    points = np.asarray(measurements, dtype=float)
    _, neighbours = cKDTree(points[:, :2]).query(points[:, :2], k=min(7, len(points)))
    local_median = np.median(points[neighbours[:, 1:], 2], axis=1)
    tolerance = max(0.02, interval * 1.25)
    return [
        tuple(item)
        for item, keep in zip(points, np.abs(points[:, 2] - local_median) <= tolerance)
        if keep
    ]


def assign_values(
    polylines: list[np.ndarray],
    readings: list[dict],
    *,
    interval: float,
    snap_tolerance: float = 0.06,
    max_label_distance: float = 70.0,
) -> dict:
    profile_leaks = []
    profile_suspects = []
    for points in polylines:
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        curvature = straightness(points)
        profile_leaks.append(curvature < 0.035 and length > 400)
        profile_suspects.append(curvature < 0.01 and length > 100)

    labels = []
    profile_labels = []
    profile_measurements = []
    profile_corridor_labels = 0
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
            if reading.get("profile_corridor"):
                measurement_match = re.fullmatch(r"([0-9]{3,4})", raw_text)
                if measurement_match and interval <= 0.05:
                    magnitude = int(measurement_match.group(1))
                    if 500 <= magnitude <= 5000:
                        profile_measurements.append(
                            (float(center[0]), float(center[1]), -magnitude / 1000.0)
                        )
                if LABEL_PATTERN.match(raw_text):
                    profile_corridor_labels += 1
                continue
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
        profile_measurements = [
            item
            for item in profile_measurements
            if band[0] - interval * 2 <= item[2] <= band[1] + interval * 2
        ]
    raw_profile_measurement_count = len(profile_measurements)
    profile_measurements = filter_profile_measurements(profile_measurements, interval)

    trees = [cKDTree(points) for points in polylines]
    line_lengths = [
        float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        for points in polylines
    ]
    minimum_contour_length = max(80.0, max_label_distance * 0.75)
    votes: dict[int, list[float]] = {}
    matched_labels = []
    label_matches = []
    unmatched = 0
    for x, y, value in labels:
        candidates = [
            (float(tree.query([x, y])[0]), index)
            for index, tree in enumerate(trees)
            if not profile_leaks[index] and line_lengths[index] >= minimum_contour_length
        ]
        distance, index = min(candidates, default=(float("inf"), -1))
        if distance <= max_label_distance:
            votes.setdefault(index, []).append(value)
            matched_labels.append((x, y, value))
            label_matches.append(
                {"x": x, "y": y, "value_km": value, "line_id": index, "distance_px": distance}
            )
        else:
            unmatched += 1
            label_matches.append(
                {"x": x, "y": y, "value_km": value, "line_id": None, "distance_px": distance}
            )

    values = {}
    confident = []
    conflicts = []
    for index, candidates in votes.items():
        counts = Counter(candidates)
        value, count = counts.most_common(1)[0]
        values[index] = value
        if (
            len(counts) == 1 or count >= 2 * (len(candidates) - count)
        ) and not profile_suspects[index]:
            confident.append(index)
        else:
            conflicts.append(index)
    return {
        "values": values,
        "confident": confident,
        "conflicts": conflicts,
        "profile_leaks": profile_leaks,
        "profile_suspects": profile_suspects,
        "labels": labels,
        "matched_labels": matched_labels,
        "label_matches": label_matches,
        "profile_labels": profile_labels,
        "profile_measurements": profile_measurements,
        "raw_profile_measurement_count": raw_profile_measurement_count,
        "profile_corridor_labels": profile_corridor_labels,
        "rejected_labels": rejected,
        "unmatched_labels": unmatched,
        "minimum_contour_length_px": minimum_contour_length,
    }


def run(
    isolines_path: Path,
    readings_path: Path,
    image_path: Path,
    output_dir: Path,
    *,
    interval: float | None = None,
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
    profile_mask_path = isolines_path.with_name("profile_mask.png")
    if profile_mask_path.exists():
        profile_mask = cv2.imdecode(
            np.fromfile(str(profile_mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if profile_mask is not None:
            # OCR boxes create a hole around the underlying profile stroke;
            # cover roughly one text height on either side of the Hough line.
            radius = max(5, round(min(profile_mask.shape) * 0.01))
            profile_mask = cv2.dilate(
                profile_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
            )
            mask_scale_x = profile_mask.shape[1] / image.shape[1]
            mask_scale_y = profile_mask.shape[0] / image.shape[0]
            for reading in readings:
                quad = np.asarray(reading.get("quad") or [], dtype=float)
                if quad.shape != (4, 2):
                    continue
                center_x, center_y = quad.mean(axis=0)
                mask_x = int(np.clip(round(center_x * mask_scale_x), 0, profile_mask.shape[1] - 1))
                mask_y = int(np.clip(round(center_y * mask_scale_y), 0, profile_mask.shape[0] - 1))
                reading["profile_corridor"] = bool(profile_mask[mask_y, mask_x])
    numeric_centers = []
    for reading in readings:
        quad = np.asarray(reading.get("quad") or [], dtype=float)
        if quad.shape != (4, 2):
            continue
        if any(
            re.fullmatch(r"-?[0-9]{3,4}", str(item.get("text", "")).strip())
            for item in reading.get("values") or []
        ):
            numeric_centers.append(quad.mean(axis=0))
    if len(numeric_centers) >= 3:
        numeric_tree = cKDTree(np.asarray(numeric_centers))
        density_radius = max(60.0, min(image.shape) * 0.02)
        for reading in readings:
            quad = np.asarray(reading.get("quad") or [], dtype=float)
            if quad.shape != (4, 2):
                continue
            center = quad.mean(axis=0)
            if len(numeric_tree.query_ball_point(center, density_radius)) >= 3:
                reading["profile_corridor"] = True
    interval_source = "provided"
    interval_inference = None
    if interval is None:
        interval, interval_inference = infer_contour_interval(readings)
        interval_source = "inferred"
    max_label_distance = max(70.0, min(image.shape) * 0.02)
    snap_tolerance = min(0.06, interval * 0.3)
    result = assign_values(
        polylines,
        readings,
        interval=interval,
        snap_tolerance=snap_tolerance,
        max_label_distance=max_label_distance,
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
    for label_index, (x, y, value) in enumerate(result["matched_labels"]):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": len(polylines) + label_index,
                    "kind": "contour_label",
                    "value_km": value,
                    "confident": True,
                    "source": "ocr_label_soft_constraint",
                },
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
        )
    point_offset = len(polylines) + len(result["matched_labels"])
    for mark_index, (x, y, value) in enumerate(result["profile_measurements"]):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": point_offset + mark_index,
                    "kind": "profile_measurement",
                    "value_km": value,
                    "confident": True,
                    "source": "ocr_profile_measurement",
                },
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
        )
    geojson_path = output_dir / "valued_contours_pixels.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    matches_path = output_dir / "label_matches.json"
    matches_path.write_text(
        json.dumps(result["label_matches"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(16, 13), dpi=120)
    preview_step = max(1, int(np.ceil(max(image.shape) / 3000.0)))
    preview_image = image[::preview_step, ::preview_step]
    ax.imshow(
        preview_image,
        cmap="gray",
        alpha=0.25,
        extent=(0, image.shape[1], image.shape[0], 0),
    )
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
        "profile_suspects": int(sum(result["profile_suspects"])),
        "ocr_labels": len(result["labels"]),
        "point_constraints": len(result["matched_labels"]),
        "matched_label_observations": len(result["matched_labels"]),
        "profile_id_labels": len(result["profile_labels"]),
        "profile_corridor_labels": result["profile_corridor_labels"],
        "profile_measurements": len(result["profile_measurements"]),
        "raw_profile_measurements": result["raw_profile_measurement_count"],
        "rejected_labels": result["rejected_labels"],
        "unmatched_labels": result["unmatched_labels"],
        "valued_polylines": len(result["values"]),
        "confident_polylines": len(result["confident"]),
        "conflicting_polylines": len(result["conflicts"]),
        "direct_value_rate": round(len(result["values"]) / max(1, len(polylines)), 4),
        "contour_interval_km": interval,
        "contour_interval_source": interval_source,
        "contour_interval_inference": interval_inference,
        "snap_tolerance_km": round(snap_tolerance, 5),
        "max_label_distance_px": round(max_label_distance, 2),
        "minimum_contour_length_px": round(result["minimum_contour_length_px"], 2),
        "ocr_image_size": [ocr_width, ocr_height],
        "trace_image_size": [image.shape[1], image.shape[0]],
        "ocr_to_trace_scale": [round(scale_x, 6), round(scale_y, 6)],
        "files": {
            "geojson": str(geojson_path),
            "preview": str(preview_path),
            "label_matches": str(matches_path),
        },
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
    parser.add_argument("--interval", type=float)
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
