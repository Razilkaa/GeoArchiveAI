"""Detect and consolidate straight seismic profile lines on a scanned map.

The output is deliberately kept in pixel coordinates. Georeferencing is a
separate stage: profile labels and the survey shapefile provide its anchors.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Segment:
    p0: tuple[float, float]
    p1: tuple[float, float]
    angle_deg: float
    rho: float
    length: float


def angle_distance(a: float, b: float) -> float:
    """Smallest angle between unoriented lines, in degrees."""
    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def segment_from_hough(values: np.ndarray) -> Segment:
    x0, y0, x1, y1 = (float(v) for v in values)
    dx, dy = x1 - x0, y1 - y0
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    theta = math.radians(angle)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    midpoint = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return Segment(
        p0=(x0, y0),
        p1=(x1, y1),
        angle_deg=angle,
        rho=float(normal @ midpoint),
        length=math.hypot(dx, dy),
    )


def _line_interval(segment: Segment, direction: np.ndarray) -> tuple[float, float]:
    values = [np.dot(segment.p0, direction), np.dot(segment.p1, direction)]
    return min(values), max(values)


def segments_match(
    left: Segment,
    right: Segment,
    *,
    angle_tolerance: float = 2.0,
    rho_tolerance: float = 18.0,
    max_gap: float = 220.0,
) -> bool:
    if angle_distance(left.angle_deg, right.angle_deg) > angle_tolerance:
        return False
    if abs(left.rho - right.rho) > rho_tolerance:
        return False
    angle = math.radians((left.angle_deg + right.angle_deg) / 2)
    direction = np.array([math.cos(angle), math.sin(angle)])
    a0, a1 = _line_interval(left, direction)
    b0, b1 = _line_interval(right, direction)
    return max(a0, b0) <= min(a1, b1) + max_gap


def consolidate_segments(segments: list[Segment]) -> list[dict]:
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i, left in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if segments_match(left, segments[j]):
                union(i, j)

    groups: dict[int, list[Segment]] = {}
    for index, segment in enumerate(segments):
        groups.setdefault(find(index), []).append(segment)

    lines = []
    for group in groups.values():
        points = np.array([point for segment in group for point in (segment.p0, segment.p1)])
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        direction = vh[0]
        if direction[0] < 0:
            direction *= -1
        projections = (points - center) @ direction
        p0 = center + projections.min() * direction
        p1 = center + projections.max() * direction
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
        lines.append(
            {
                "p0": p0.round(2).tolist(),
                "p1": p1.round(2).tolist(),
                "angle_deg": round(angle, 2),
                "length_px": round(float(projections.max() - projections.min()), 2),
                "support": len(group),
            }
        )
    lines.sort(key=lambda item: (-item["length_px"], -item["support"]))
    for index, line in enumerate(lines):
        line["id"] = index
    return lines


def detect_lines(image: np.ndarray) -> tuple[list[Segment], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    long_ink = np.zeros_like(ink)
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if max(width, height) >= 300 and area >= 250:
            long_ink[labels == component] = 255

    raw = cv2.HoughLinesP(
        long_ink,
        rho=1,
        theta=np.pi / 720,
        threshold=160,
        minLineLength=350,
        maxLineGap=50,
    )
    segments = [] if raw is None else [segment_from_hough(row.reshape(4)) for row in raw]
    return segments, long_ink


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    try:
        image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    segments, _ = detect_lines(image)
    lines = consolidate_segments(segments)

    height, width = image.shape[:2]
    margin = 90
    map_bottom = int(height * 0.84)
    lines = [
        line
        for line in lines
        if (line["p0"][0] + line["p1"][0]) / 2 > 180
        and (line["p0"][0] + line["p1"][0]) / 2 < width - 180
        and (line["p0"][1] + line["p1"][1]) / 2 > 180
        and (line["p0"][1] + line["p1"][1]) / 2 < map_bottom
        if line["length_px"] >= 450
        and all(margin < point[0] < width - margin for point in (line["p0"], line["p1"]))
        and min(line["p0"][1], line["p1"][1]) < map_bottom
        and not (
            angle_distance(line["angle_deg"], 0) < 1
            and (min(line["p0"][1], line["p1"][1]) < 180 or max(line["p0"][1], line["p1"][1]) > map_bottom)
        )
    ]
    for index, line in enumerate(lines):
        line["id"] = index

    payload = {
        "source": str(args.image),
        "image_size": [width, height],
        "method": "connected-components+hough+line-clustering",
        "raw_segments": len(segments),
        "profile_candidates": lines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.preview:
        preview = image.copy()
        for line in lines:
            p0 = tuple(round(v) for v in line["p0"])
            p1 = tuple(round(v) for v in line["p1"])
            cv2.line(preview, p0, p1, (0, 0, 255), 6, cv2.LINE_AA)
            midpoint = tuple(round((a + b) / 2) for a, b in zip(p0, p1))
            cv2.putText(preview, str(line["id"]), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
        cv2.imwrite(str(args.preview), preview)
    print(f"{len(segments)} Hough segments -> {len(lines)} profile candidates")


if __name__ == "__main__":
    main()
