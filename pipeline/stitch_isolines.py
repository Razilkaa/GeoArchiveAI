from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from trace_map_isolines import imread_gray, imwrite, skeletonize, vectorize_skeleton


NEIGHBOURS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


def read_isoline_labels(path: Path, scale: float) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = []
    for item in payload.get("items", []):
        reading = item.get("reading", {})
        text = str(reading.get("text") or "").replace(",", ".")
        if not reading.get("readable"):
            continue
        if not re.fullmatch(r"-?\d\.[02468]0", text):
            continue
        x, y, width, height = item["bbox"]
        # Font-band classification is only a detector hint. Older maps often use
        # smaller or rotated contour labels, so retain sufficiently large strict
        # d.d0 labels and let gap topology decide whether they anchor a line.
        if item.get("font_band") != "big" and width * height < 5000:
            continue
        value = float(text)
        if not 0.6 <= value <= 3.0:
            continue
        labels.append(
            {
                "id": int(item["id"]),
                "value": value,
                "text": text,
                "bbox": [x * scale, y * scale, width * scale, height * scale],
                "center": [(x + width / 2) * scale, (y + height / 2) * scale],
            }
        )
    return labels


def pixel_set(skeleton: np.ndarray) -> set[tuple[int, int]]:
    return {(int(x), int(y)) for y, x in np.argwhere(skeleton > 0)}


def neighbours(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = point
    return [(x + dx, y + dy) for dx, dy in NEIGHBOURS if (x + dx, y + dy) in pixels]


def endpoint_tangent(
    endpoint: tuple[int, int], pixels: set[tuple[int, int]], steps: int = 12
) -> np.ndarray | None:
    path = [endpoint]
    previous = None
    current = endpoint
    for _ in range(steps):
        candidates = [point for point in neighbours(current, pixels) if point != previous]
        if not candidates:
            break
        if len(candidates) > 1:
            if previous is None:
                following = candidates[0]
            else:
                incoming = np.array(current) - np.array(previous)
                following = max(candidates, key=lambda point: float(np.dot(incoming, np.array(point) - np.array(current))))
        else:
            following = candidates[0]
        path.append(following)
        previous, current = current, following
    if len(path) < 3:
        return None
    outward = np.array(path[0], dtype=float) - np.array(path[-1], dtype=float)
    norm = float(np.linalg.norm(outward))
    return outward / norm if norm else None


def endpoints_with_tangents(skeleton: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    pixels = pixel_set(skeleton)
    result = {}
    for point in pixels:
        if len(neighbours(point, pixels)) == 1:
            tangent = endpoint_tangent(point, pixels)
            if tangent is not None:
                result[point] = tangent
    return result


def pair_score(
    first: tuple[int, int],
    second: tuple[int, int],
    tangents: dict[tuple[int, int], np.ndarray],
    max_distance: float,
) -> float | None:
    vector = np.array(second, dtype=float) - np.array(first, dtype=float)
    distance = float(np.linalg.norm(vector))
    if distance < 3 or distance > max_distance:
        return None
    direction = vector / distance
    first_alignment = float(np.dot(tangents[first], direction))
    second_alignment = float(np.dot(tangents[second], -direction))
    tangent_alignment = float(np.dot(tangents[first], -tangents[second]))
    if first_alignment < 0.68 or second_alignment < 0.68 or tangent_alignment < 0.55:
        return None
    return distance + 20 * (1 - first_alignment) + 20 * (1 - second_alignment) + 8 * (1 - tangent_alignment)


def draw_bridge(
    mask: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    first_tangent: np.ndarray,
    second_tangent: np.ndarray,
) -> None:
    p0 = np.array(first, dtype=float)
    p1 = np.array(second, dtype=float)
    distance = float(np.linalg.norm(p1 - p0))
    m0 = first_tangent * distance * 0.38
    m1 = -second_tangent * distance * 0.38
    points = []
    for value in np.linspace(0, 1, max(8, int(distance * 1.5))):
        t2, t3 = value * value, value * value * value
        point = (
            (2 * t3 - 3 * t2 + 1) * p0
            + (t3 - 2 * t2 + value) * m0
            + (-2 * t3 + 3 * t2) * p1
            + (t3 - t2) * m1
        )
        points.append(np.rint(point).astype(np.int32))
    cv2.polylines(mask, [np.array(points)], False, 255, 1, cv2.LINE_8)


def label_bridges(
    skeleton: np.ndarray,
    labels: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    bridge_mask = np.zeros_like(skeleton)
    used = set()
    bridge_records = []
    tangents = endpoints_with_tangents(skeleton)
    components, component_meta, _ = assign_component_values(skeleton, labels)
    for label in labels:
        x, y, width, height = label["bbox"]
        margin = max(14.0, max(width, height) * 0.65)
        candidates = [
            point
            for point in tangents
            if x - margin <= point[0] <= x + width + margin
            and y - margin <= point[1] <= y + height + margin
            and point not in used
        ]
        pairs = []
        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                first_values = set(component_meta.get(int(components[first[1], first[0]]), {}).get("label_values", []))
                second_values = set(component_meta.get(int(components[second[1], second[0]]), {}).get("label_values", []))
                if any(value != label["value"] for value in first_values | second_values):
                    continue
                score = pair_score(first, second, tangents, max_distance=math.hypot(width, height) + 2 * margin)
                if score is None:
                    continue
                midpoint = (np.array(first) + np.array(second)) / 2
                center_distance = float(np.linalg.norm(midpoint - np.array(label["center"])))
                pairs.append((score + center_distance * 0.35, first, second))
        if not pairs:
            continue
        _, first, second = min(pairs, key=lambda value: value[0])
        draw_bridge(bridge_mask, first, second, tangents[first], tangents[second])
        used.update({first, second})
        bridge_records.append({"kind": "label_gap", "label_id": label["id"], "value": label["value"], "from": first, "to": second})
    return bridge_mask, bridge_records


def generic_bridges(
    skeleton: np.ndarray,
    labels: list[dict[str, Any]],
    max_distance: float = 32.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tangents = endpoints_with_tangents(skeleton)
    points = list(tangents)
    if not points:
        return np.zeros_like(skeleton), []
    tree = cKDTree(np.array(points))
    components, component_meta, _ = assign_component_values(skeleton, labels)
    options = []
    for index, first in enumerate(points):
        for other_index in tree.query_ball_point(first, max_distance):
            if other_index <= index:
                continue
            second = points[other_index]
            first_values = set(component_meta.get(int(components[first[1], first[0]]), {}).get("label_values", []))
            second_values = set(component_meta.get(int(components[second[1], second[0]]), {}).get("label_values", []))
            if first_values and second_values and first_values != second_values:
                continue
            score = pair_score(first, second, tangents, max_distance)
            if score is not None:
                options.append((score, first, second))
    used = set()
    bridge_mask = np.zeros_like(skeleton)
    records = []
    for score, first, second in sorted(options, key=lambda value: value[0]):
        if first in used or second in used:
            continue
        draw_bridge(bridge_mask, first, second, tangents[first], tangents[second])
        used.update({first, second})
        records.append({"kind": "short_gap", "score": round(score, 2), "from": first, "to": second})
    return bridge_mask, records


def assign_component_values(
    skeleton: np.ndarray, labels: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[int, dict[str, Any]], list[dict[str, Any]]]:
    count, components = cv2.connectedComponents(np.uint8(skeleton > 0), connectivity=8)
    yx = np.argwhere(skeleton > 0)
    xy = np.column_stack([yx[:, 1], yx[:, 0]])
    tree = cKDTree(xy)
    assignments = []
    values_by_component: dict[int, list[float]] = {}
    for label in labels:
        distance, index = tree.query(label["center"])
        x, y = xy[int(index)]
        component = int(components[int(y), int(x)])
        if distance > 70 or component == 0:
            continue
        values_by_component.setdefault(component, []).append(label["value"])
        assignments.append({**label, "component": component, "distance_px": round(float(distance), 2)})
    metadata = {}
    for component in range(1, count):
        values = sorted(set(values_by_component.get(component, [])))
        metadata[component] = {
            "value": values[0] if len(values) == 1 else None,
            "label_values": values,
            "conflict": len(values) > 1,
        }
    return components, metadata, assignments


def merge_polylines_by_continuity(
    polylines: list[list[list[float]]],
    max_distance: float = 22.0,
    min_opposition: float = 0.72,
) -> list[list[list[float]]]:
    """Join graph fragments through crossings by choosing the straightest continuation."""
    lines = [np.asarray(line, dtype=float) for line in polylines if len(line) >= 2]

    def outward_tangent(line: np.ndarray, at_start: bool) -> np.ndarray | None:
        sample = min(4, len(line) - 1)
        vector = line[0] - line[sample] if at_start else line[-1] - line[-1 - sample]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else None

    while True:
        options = []
        for first_index, first in enumerate(lines):
            for second_index in range(first_index + 1, len(lines)):
                second = lines[second_index]
                for first_start in (True, False):
                    first_point = first[0] if first_start else first[-1]
                    first_tangent = outward_tangent(first, first_start)
                    if first_tangent is None:
                        continue
                    for second_start in (True, False):
                        second_point = second[0] if second_start else second[-1]
                        distance = float(np.linalg.norm(second_point - first_point))
                        if distance > max_distance:
                            continue
                        second_tangent = outward_tangent(second, second_start)
                        if second_tangent is None:
                            continue
                        opposition = -float(np.dot(first_tangent, second_tangent))
                        if opposition < min_opposition:
                            continue
                        options.append(
                            (
                                distance + 18.0 * (1.0 - opposition),
                                first_index,
                                second_index,
                                first_start,
                                second_start,
                            )
                        )
        if not options:
            break
        _, first_index, second_index, first_start, second_start = min(options)
        first = lines[first_index]
        second = lines[second_index]
        if first_start:
            first = first[::-1]
        if not second_start:
            second = second[::-1]
        if np.linalg.norm(first[-1] - second[0]) < 1.0:
            merged = np.vstack([first, second[1:]])
        else:
            merged = np.vstack([first, second])
        lines[first_index] = merged
        del lines[second_index]

    return [
        [[round(float(x), 1), round(float(y), 1)] for x, y in line]
        for line in lines
    ]


def propagate_values_by_tangent(features: list[dict[str, Any]], rounds: int = 4) -> int:
    """Carry a trusted value across a remaining gap without crossing contours."""

    def endpoint_data(feature: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
        line = np.asarray(feature["geometry"]["coordinates"], dtype=float)
        if len(line) < 2:
            return []
        sample = min(4, len(line) - 1)
        result = []
        for point, inner in ((line[0], line[sample]), (line[-1], line[-1 - sample])):
            tangent = point - inner
            norm = float(np.linalg.norm(tangent))
            if norm:
                result.append((point, tangent / norm))
        return result

    endpoint_cache = [endpoint_data(feature) for feature in features]
    propagated = 0
    for round_index in range(rounds):
        proposals: list[tuple[int, float, float]] = []
        valued = [
            index
            for index, feature in enumerate(features)
            if feature["properties"]["value"] is not None
            and not feature["properties"]["value_conflict"]
        ]
        for index, feature in enumerate(features):
            properties = feature["properties"]
            if properties["value"] is not None or properties["length_px"] < 20:
                continue
            candidates = []
            for other_index in valued:
                if other_index == index:
                    continue
                other = features[other_index]
                if other["properties"]["length_px"] < 30:
                    continue
                for point, tangent in endpoint_cache[index]:
                    for other_point, other_tangent in endpoint_cache[other_index]:
                        delta = other_point - point
                        distance = float(np.linalg.norm(delta))
                        if distance > 180:
                            continue
                        opposition = -float(np.dot(tangent, other_tangent))
                        if distance <= 24:
                            if opposition < 0.7:
                                continue
                            score = distance + 24 * (1 - opposition)
                        else:
                            direction = delta / distance
                            first_alignment = float(np.dot(tangent, direction))
                            second_alignment = float(np.dot(other_tangent, -direction))
                            if first_alignment < 0.72 or second_alignment < 0.72 or opposition < 0.58:
                                continue
                            score = distance + 30 * (2 - first_alignment - second_alignment) + 18 * (1 - opposition)
                        candidates.append((score, float(other["properties"]["value"])))
            if not candidates:
                continue
            candidates.sort()
            best_score = candidates[0][0]
            nearby_values = {value for score, value in candidates if score <= best_score + 24}
            if len(nearby_values) == 1:
                proposals.append((index, nearby_values.pop(), best_score))
        if not proposals:
            break
        for index, value, score in proposals:
            properties = features[index]["properties"]
            if properties["value"] is not None:
                continue
            properties["value"] = value
            properties["value_source"] = "tangent_gap_propagation"
            properties["propagation_round"] = round_index + 1
            properties["propagation_score"] = round(float(score), 2)
            propagated += 1
    return propagated


def run(
    source: Path,
    isoline_mask_path: Path,
    readings_path: Path,
    output_dir: Path,
    scale: float = 0.25,
    max_gap: float = 32.0,
) -> Path:
    mask = imread_gray(isoline_mask_path)
    raw_skeleton = skeletonize(mask)
    labels = read_isoline_labels(readings_path, scale)
    endpoints_before = len(endpoints_with_tangents(raw_skeleton))

    label_mask, label_records = label_bridges(raw_skeleton, labels)
    after_labels = skeletonize(cv2.bitwise_or(raw_skeleton, label_mask))
    short_mask, short_records = generic_bridges(after_labels, labels, max_distance=max_gap)
    bridge_mask = cv2.bitwise_or(label_mask, short_mask)
    stitched = skeletonize(cv2.bitwise_or(after_labels, short_mask))
    endpoints_after = len(endpoints_with_tangents(stitched))

    components, component_meta, assignments = assign_component_values(stitched, labels)
    raw_polylines = vectorize_skeleton(stitched, scale, min_pixels=8)
    polylines = merge_polylines_by_continuity(
        raw_polylines,
        max_distance=36.0,
        min_opposition=0.58,
    )
    line_values: dict[int, list[float]] = {}
    direct_line_assignments = []
    if polylines:
        vertices = []
        owners = []
        for line_index, line in enumerate(polylines):
            vertices.extend(line)
            owners.extend([line_index] * len(line))
        line_tree = cKDTree(np.array(vertices))
        for label in labels:
            original_center = np.array(label["center"]) / scale
            distance, vertex_index = line_tree.query(original_center)
            if distance > 260:
                continue
            line_index = owners[int(vertex_index)]
            line_values.setdefault(line_index, []).append(label["value"])
            direct_line_assignments.append(
                {**label, "line": line_index, "distance_original_px": round(float(distance), 2)}
            )
    features = []
    for index, line in enumerate(polylines):
        first_x = min(stitched.shape[1] - 1, max(0, int(round(line[0][0] * scale))))
        first_y = min(stitched.shape[0] - 1, max(0, int(round(line[0][1] * scale))))
        component = int(components[first_y, first_x])
        meta = component_meta.get(component, {"value": None, "label_values": [], "conflict": False})
        direct_values = sorted(set(line_values.get(index, [])))
        if len(direct_values) == 1:
            value = direct_values[0]
            value_source = "nearest_label"
            value_conflict = False
        elif len(direct_values) > 1:
            value = None
            value_source = "conflicting_nearest_labels"
            value_conflict = True
        elif not meta["conflict"] and meta["value"] is not None:
            value = meta["value"]
            value_source = "unambiguous_connected_component"
            value_conflict = False
        else:
            value = None
            value_source = "unassigned"
            value_conflict = False
        length_px = round(
            sum(
                math.hypot(second[0] - first[0], second[1] - first[1])
                for first, second in zip(line[:-1], line[1:])
            ),
            1,
        )
        # Keep short attributed fragments too: intersections and surviving text
        # split a valid raster contour into several pieces. Thirty source pixels
        # still provide usable geometry while removing tiny OCR remnants.
        work_accepted = value is not None and not value_conflict and length_px >= 30
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line},
                "properties": {
                    "id": index,
                    "component": component,
                    "value": value,
                    "value_source": value_source,
                    "direct_label_values": direct_values,
                    "component_label_values": meta["label_values"],
                    "value_conflict": value_conflict,
                    "length_px": length_px,
                    "work_accepted": work_accepted,
                    "geometry_source": "raster_trace_with_topology_stitching",
                    "coordinate_system": "image_pixels",
                },
            }
        )
    propagated_polylines = propagate_values_by_tangent(features)
    for feature in features:
        properties = feature["properties"]
        properties["work_accepted"] = (
            properties["value"] is not None
            and not properties["value_conflict"]
            and properties["length_px"] >= 30
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = output_dir / "stitched_isolines.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    work_features = [feature for feature in features if feature["properties"]["work_accepted"]]
    work_geojson_path = output_dir / "work_isolines.geojson"
    work_geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": work_features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    grid_map_path = output_dir / "digitized_grid_map.png"
    grid_source = imread_gray(source)
    fig, axis = plt.subplots(figsize=(15, 10), dpi=160)
    axis.set_facecolor("white")
    work_values = [float(feature["properties"]["value"]) for feature in work_features]
    grid_minimum = min(work_values) if work_values else 0.0
    grid_maximum = max(work_values) if work_values else 1.0
    normalizer = Normalize(vmin=grid_minimum, vmax=grid_maximum)
    colormap = plt.get_cmap("turbo")
    for feature in features:
        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
        properties = feature["properties"]
        if properties["work_accepted"]:
            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color=colormap(normalizer(float(properties["value"]))),
                linewidth=1.8,
                solid_capstyle="round",
                zorder=2,
            )
        else:
            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="#aeb7bf",
                linewidth=1.0,
                alpha=0.9,
                zorder=1,
            )
    longest_by_value: dict[float, dict[str, Any]] = {}
    for feature in work_features:
        value = float(feature["properties"]["value"])
        current = longest_by_value.get(value)
        if current is None or feature["properties"]["length_px"] > current["properties"]["length_px"]:
            longest_by_value[value] = feature
    for value, feature in sorted(longest_by_value.items()):
        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
        anchor = coordinates[len(coordinates) // 2]
        axis.annotate(
            f"{value:.2f}",
            xy=(anchor[0], anchor[1]),
            xytext=(4, -4),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            color="#15191d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=3,
        )
    all_grid_coordinates = np.vstack(
        [np.asarray(feature["geometry"]["coordinates"], dtype=float) for feature in features]
    )
    padding = max(grid_source.shape) * 0.025
    axis.set_xlim(
        max(0, float(all_grid_coordinates[:, 0].min() - padding)),
        min(grid_source.shape[1], float(all_grid_coordinates[:, 0].max() + padding)),
    )
    axis.set_ylim(
        min(grid_source.shape[0], float(all_grid_coordinates[:, 1].max() + padding)),
        max(0, float(all_grid_coordinates[:, 1].min() - padding)),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#dce1e5", linewidth=0.55)
    axis.set_xlabel("X, пиксели исходного растра")
    axis.set_ylabel("Y, пиксели исходного растра")
    axis.set_title("Оцифрованные изолинии, лист 22 (без геопривязки)", fontsize=13)
    scalar = ScalarMappable(norm=normalizer, cmap=colormap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Значение изолинии")
    axis.text(
        0.01,
        0.01,
        "Цвет: атрибутированные линии   Серый: геометрия без надёжного значения",
        transform=axis.transAxes,
        fontsize=8,
        color="#4a5157",
        bbox={"facecolor": "white", "edgecolor": "#d2d6da", "alpha": 0.9, "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(grid_map_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    original = imread_gray(source)
    small = cv2.resize(original, (stitched.shape[1], stitched.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    overlay[stitched > 0] = (0, 0, 220)
    overlay[bridge_mask > 0] = (220, 0, 220)
    imwrite(output_dir / "stitched_isoline_mask.png", stitched)
    imwrite(output_dir / "bridge_mask.png", bridge_mask)
    imwrite(output_dir / "stitched_isolines_overlay.png", overlay)
    attributed = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    values = [feature["properties"]["value"] for feature in work_features]
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 1.0
    labeled_keys = set()
    for feature in features:
        coordinates = np.array(feature["geometry"]["coordinates"], dtype=float) * scale
        points = np.rint(coordinates).astype(np.int32)
        properties = feature["properties"]
        if properties["work_accepted"]:
            ratio = (properties["value"] - minimum) / max(1e-6, maximum - minimum)
            color = cv2.applyColorMap(np.uint8([[round(ratio * 255)]]), cv2.COLORMAP_TURBO)[0, 0]
            color_tuple = tuple(int(channel) for channel in color)
            cv2.polylines(attributed, [points], False, color_tuple, 3, cv2.LINE_AA)
            key = (properties["component"], properties["value"])
            if key not in labeled_keys and len(points):
                anchor = points[len(points) // 2]
                cv2.putText(
                    attributed,
                    f"{properties['value']:.2f}",
                    (int(anchor[0]) + 4, int(anchor[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
                labeled_keys.add(key)
        else:
            cv2.polylines(attributed, [points], False, (170, 170, 170), 1, cv2.LINE_AA)
    imwrite(output_dir / "attributed_isolines_overlay.png", attributed)

    labeled_components = {item["component"] for item in assignments}
    conflicts = sum(1 for value in component_meta.values() if value["conflict"])
    chain_conflicts = sum(1 for values in line_values.values() if len(set(values)) > 1)
    valued_polylines = sum(1 for feature in features if feature["properties"]["value"] is not None)
    work_polylines = len(work_features)
    summary = {
        "source": str(source),
        "coordinate_system": "image_pixels",
        "georeferenced": False,
        "endpoints_before": endpoints_before,
        "endpoints_after": endpoints_after,
        "label_gap_bridges": len(label_records),
        "short_gap_bridges": len(short_records),
        "max_gap_scaled_px": max_gap,
        "components": int(components.max()),
        "raw_vector_fragments": len(raw_polylines),
        "vector_polylines": len(polylines),
        "value_labels": len(labels),
        "assigned_labels": len(assignments),
        "labeled_components": len(labeled_components),
        "component_value_conflicts": conflicts,
        "chain_value_conflicts": chain_conflicts,
        "valued_polylines": valued_polylines,
        "propagated_polylines": propagated_polylines,
        "work_polylines": work_polylines,
        "label_assignments": assignments,
        "direct_line_assignments": direct_line_assignments,
        "status": "topology_candidate_requires_visual_qc",
        "overlay": str(output_dir / "stitched_isolines_overlay.png"),
        "attributed_overlay": str(output_dir / "attributed_isolines_overlay.png"),
        "grid_map": str(grid_map_path),
        "vector_geojson": str(geojson_path),
        "work_vector_geojson": str(work_geojson_path),
        "bridge_records": label_records + short_records,
    }
    output = output_dir / "stitching_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch raster isoline gaps using label coordinates and tangent continuity")
    parser.add_argument("source", type=Path)
    parser.add_argument("isoline_mask", type=Path)
    parser.add_argument("readings", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--max-gap", type=float, default=32.0)
    args = parser.parse_args()
    print(run(args.source, args.isoline_mask, args.readings, args.out, args.scale, args.max_gap))


if __name__ == "__main__":
    main()
