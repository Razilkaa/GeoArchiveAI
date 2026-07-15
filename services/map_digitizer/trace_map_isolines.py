from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def imread_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def load_boxes(manifest: Path) -> list[list[int]]:
    return [
        json.loads(line)["bbox"]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def skeletonize(binary: np.ndarray) -> np.ndarray:
    image = np.uint8(binary > 0) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton


def vectorize_skeleton(skeleton: np.ndarray, scale: float, min_pixels: int = 8) -> list[list[list[float]]]:
    pixels = {(int(x), int(y)) for y, x in np.argwhere(skeleton > 0)}

    def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = point
        return [
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy) and (x + dx, y + dy) in pixels
        ]

    degree = {point: len(neighbours(point)) for point in pixels}
    nodes = {point for point, count in degree.items() if count != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    chains = []

    def edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    def walk(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        chain = [start, nxt]
        visited_edges.add(edge(start, nxt))
        previous, current = start, nxt
        while current not in nodes:
            candidates = [point for point in neighbours(current) if point != previous]
            if not candidates:
                break
            following = candidates[0]
            if edge(current, following) in visited_edges:
                break
            chain.append(following)
            visited_edges.add(edge(current, following))
            previous, current = current, following
        return chain

    for start in nodes:
        for nxt in neighbours(start):
            if edge(start, nxt) not in visited_edges:
                chains.append(walk(start, nxt))
    for start in pixels:
        for nxt in neighbours(start):
            if edge(start, nxt) not in visited_edges:
                chains.append(walk(start, nxt))

    polylines = []
    for chain in chains:
        if len(chain) < min_pixels:
            continue
        points = np.array(chain, dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(points, epsilon=1.25, closed=False).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        polylines.append([[round(float(x / scale), 1), round(float(y / scale), 1)] for x, y in simplified])
    return polylines


def line_labels(readings_path: Path | None) -> list[dict]:
    if not readings_path or not readings_path.exists():
        return []
    payload = json.loads(readings_path.read_text(encoding="utf-8"))
    labels = []
    for item in payload.get("items", []):
        reading = item.get("reading", {})
        value = reading.get("parsed_value")
        text = str(reading.get("text") or "")
        if item.get("font_band") != "big" or value is None or not reading.get("readable"):
            continue
        if not re.fullmatch(r"-?\d\.[02468]0", text.replace(",", ".")):
            continue
        x, y, width, height = item["bbox"]
        labels.append({"id": item["id"], "value": float(value), "text": text, "x": x + width / 2, "y": y + height / 2})
    return labels


def trace(
    source: Path,
    manifest: Path,
    output_dir: Path,
    scale: float = 0.25,
    readings_path: Path | None = None,
) -> Path:
    original = imread_gray(source)
    height, width = original.shape
    small = cv2.resize(original, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    scaled_boxes = []
    for x, y, box_width, box_height in load_boxes(manifest):
        margin = 5
        scaled_boxes.append(
            (
                max(0, int((x - margin) * scale)),
                max(0, int((y - margin) * scale)),
                min(ink.shape[1], int((x + box_width + margin) * scale)),
                min(ink.shape[0], int((y + box_height + margin) * scale)),
            )
        )
    text_mask = np.zeros_like(ink)
    component_count, component_labels, component_stats, component_centers = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )
    for index in range(1, component_count):
        x, y, component_width, component_height, area = component_stats[index]
        center_x, center_y = component_centers[index]
        if max(component_width, component_height) > 34 or area > 520:
            continue
        if any(x0 <= center_x <= x1 and y0 <= center_y <= y1 for x0, y0, x1, y1 in scaled_boxes):
            text_mask[component_labels == index] = 255
    linework = cv2.bitwise_and(ink, cv2.bitwise_not(text_mask))

    profile_mask = np.zeros_like(linework)
    lines = cv2.HoughLinesP(
        linework,
        rho=1,
        theta=np.pi / 720,
        threshold=120,
        minLineLength=max(180, int(min(linework.shape) * 0.12)),
        maxLineGap=28,
    )
    profile_segments = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min(linework.shape) * 0.14:
                continue
            cv2.line(profile_mask, (x1, y1), (x2, y2), 255, 5)
            profile_segments += 1

    residual = cv2.bitwise_and(linework, cv2.bitwise_not(profile_mask))
    frame = max(10, int(min(residual.shape) * 0.02))
    residual[:frame, :] = 0
    residual[-frame:, :] = 0
    residual[:, :frame] = 0
    residual[:, -frame:] = 0
    residual[int(residual.shape[0] * 0.82):, :] = 0

    residual = cv2.morphologyEx(
        residual,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    isolines = np.zeros_like(residual)
    kept_components = 0
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        span = max(component_width, component_height)
        if area < 22 or span < 35:
            continue
        component = np.uint8(labels == index) * 255
        isolines = cv2.bitwise_or(isolines, component)
        kept_components += 1

    preview = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    preview[profile_mask > 0] = (255, 160, 0)
    preview[isolines > 0] = (0, 0, 220)
    imwrite(output_dir / "profile_mask.png", profile_mask)
    imwrite(output_dir / "isoline_mask.png", isolines)
    imwrite(output_dir / "traced_linework_overlay.png", preview)
    skeleton = skeletonize(isolines)
    polylines = vectorize_skeleton(skeleton, scale)
    labels = line_labels(readings_path)
    assignments: dict[int, list[dict]] = {}
    if polylines and labels:
        vertices = []
        owners = []
        for index, line in enumerate(polylines):
            vertices.extend(line)
            owners.extend([index] * len(line))
        tree = cKDTree(np.array(vertices))
        for label in labels:
            distance, vertex_index = tree.query([label["x"], label["y"]])
            if distance <= 240:
                assignments.setdefault(owners[int(vertex_index)], []).append({**label, "distance_px": round(float(distance), 1)})
    features = []
    for index, line in enumerate(polylines):
        anchors = assignments.get(index, [])
        values = sorted({anchor["value"] for anchor in anchors})
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line},
                "properties": {
                    "id": index,
                    "value": values[0] if len(values) == 1 else None,
                    "label_values": values,
                    "label_anchors": anchors,
                    "geometry_source": "raster_trace",
                    "coordinate_system": "image_pixels",
                },
            }
        )
    vector_path = output_dir / "traced_isolines.geojson"
    vector_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "manifest": str(manifest),
        "scale": scale,
        "profile_segments_removed": profile_segments,
        "isoline_components": kept_components,
        "vector_polylines": len(polylines),
        "isoline_value_labels": len(labels),
        "assigned_value_labels": sum(len(value) for value in assignments.values()),
        "status": "geometry_candidate_requires_visual_qc",
        "overlay": str(output_dir / "traced_linework_overlay.png"),
        "isoline_mask": str(output_dir / "isoline_mask.png"),
        "profile_mask": str(output_dir / "profile_mask.png"),
        "vector_geojson": str(vector_path),
    }
    output = output_dir / "tracing_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace raster isoline geometry separately from profiles and labels")
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--readings", type=Path)
    args = parser.parse_args()
    print(trace(args.source, args.manifest, args.out, args.scale, args.readings))


if __name__ == "__main__":
    main()
