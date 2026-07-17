"""Trace long authored contours from archival map scans.

This is the production form of the original sheet-23 tracer.  It deliberately
works at source resolution: small connected components are text/ticks, long
straight components are profiles/frame, and the remaining skeleton is joined
through gaps only when endpoint tangents agree.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def _walk_skeleton(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    pixels = set(map(tuple, np.argwhere(skeleton)))
    offsets = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )

    def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = point
        return [(y + dy, x + dx) for dy, dx in offsets if (y + dy, x + dx) in pixels]

    degree = {point: len(neighbours(point)) for point in pixels}
    nodes = {point for point, count in degree.items() if count != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    def edge(left: tuple[int, int], right: tuple[int, int]):
        return (left, right) if left <= right else (right, left)

    def walk(start: tuple[int, int], first: tuple[int, int]):
        path = [start, first]
        visited_edges.add(edge(start, first))
        previous, current = start, first
        while current not in nodes:
            candidates = [item for item in neighbours(current) if item != previous]
            if not candidates:
                break
            following = candidates[0]
            if edge(current, following) in visited_edges:
                break
            path.append(following)
            visited_edges.add(edge(current, following))
            previous, current = current, following
        return path

    for node in nodes:
        for neighbour in neighbours(node):
            if edge(node, neighbour) not in visited_edges:
                paths.append(walk(node, neighbour))
    for point in pixels:
        for neighbour in neighbours(point):
            if edge(point, neighbour) not in visited_edges:
                paths.append(walk(point, neighbour))
    return paths


def _stitch(
    paths: list[np.ndarray], *, max_gap: float, minimum_alignment: float
) -> list[np.ndarray]:
    lines: list[np.ndarray | None] = list(paths)

    def tangent(line: np.ndarray, at_start: bool, count: int = 12) -> np.ndarray:
        segment = line[:count] if at_start else line[-count:][::-1]
        vector = segment[0] - segment[-1]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    changed = True
    while changed:
        changed = False
        endpoints = []
        for index, line in enumerate(lines):
            if line is None:
                continue
            endpoints.extend(((line[0], index, True), (line[-1], index, False)))
        if len(endpoints) < 2:
            break
        tree = cKDTree(np.asarray([item[0] for item in endpoints]))
        used: set[int] = set()
        for left_index, right_index in sorted(tree.query_pairs(max_gap)):
            if left_index in used or right_index in used:
                continue
            point_a, line_a, start_a = endpoints[left_index]
            point_b, line_b, start_b = endpoints[right_index]
            if line_a == line_b or lines[line_a] is None or lines[line_b] is None:
                continue
            first = lines[line_a]
            second = lines[line_b]
            assert first is not None and second is not None
            bridge = point_b - point_a
            distance = float(np.linalg.norm(bridge))
            tangent_a = tangent(first, start_a)
            tangent_b = tangent(second, start_b)
            continuation = float(np.dot(tangent_a, -tangent_b))
            if continuation < minimum_alignment:
                continue
            if distance > 3.0:
                direction = bridge / distance
                if min(float(np.dot(tangent_a, direction)), float(np.dot(-tangent_b, direction))) < minimum_alignment:
                    continue
            joined_a = first[::-1] if start_a else first
            joined_b = second if start_b else second[::-1]
            lines[line_a] = np.vstack([joined_a, joined_b])
            lines[line_b] = None
            used.update((left_index, right_index))
            changed = True
    return [line for line in lines if line is not None]


def trace_source_geometry(gray: np.ndarray) -> dict:
    ink = np.uint8(gray < 128)
    height, width = ink.shape
    resolution_scale = max(0.5, min(height, width) / 8_000.0)
    min_component_span = max(55, round(110 * resolution_scale))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    retained = np.zeros_like(ink)
    dashes = []
    for component in range(1, count):
        x, y, component_width, component_height, area = stats[component]
        major = max(component_width, component_height)
        minor = min(component_width, component_height)
        if major >= min_component_span:
            retained[labels == component] = 1
            continue
        if not (max(7, 14 * resolution_scale) <= major <= 150 * resolution_scale):
            continue
        if area < max(6, 12 * resolution_scale) or major / max(1, minor) < 2.2:
            continue
        ys, xs = np.nonzero(labels[y:y + component_height, x:x + component_width] == component)
        points = np.column_stack([xs + x, ys + y]).astype(float)
        mean = points.mean(axis=0)
        _, singular, vectors = np.linalg.svd(points - mean)
        if singular[1] / max(1e-3, singular[0]) < 0.35:
            dashes.append((mean, vectors[0], major / 2.0, component))

    dash_links = 0
    if dashes:
        centers = np.asarray([item[0] for item in dashes])
        directions = np.asarray([item[1] for item in dashes])
        half_lengths = np.asarray([item[2] for item in dashes])
        tree = cKDTree(centers)
        for left, right in tree.query_pairs(80 * resolution_scale):
            delta = centers[right] - centers[left]
            distance = float(np.linalg.norm(delta))
            if distance < 4 or distance > half_lengths[left] + half_lengths[right] + 65 * resolution_scale:
                continue
            direction = delta / distance
            if min(abs(float(np.dot(direction, directions[left]))), abs(float(np.dot(direction, directions[right])))) < 0.9:
                continue
            cv2.line(retained, tuple(centers[left].astype(int)), tuple(centers[right].astype(int)), 1, max(1, round(3 * resolution_scale)))
            retained[labels == dashes[left][3]] = 1
            retained[labels == dashes[right][3]] = 1
            dash_links += 1

    profile_mask = np.zeros_like(retained)
    segments = cv2.HoughLinesP(
        retained * 255,
        1,
        np.pi / 360,
        threshold=max(150, round(300 * resolution_scale)),
        minLineLength=max(250, round(500 * resolution_scale)),
        maxLineGap=max(6, round(12 * resolution_scale)),
    )
    segment_count = 0
    if segments is not None:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            cv2.line(profile_mask, (x1, y1), (x2, y2), 1, max(3, round(9 * resolution_scale)))
            segment_count += 1
    retained[profile_mask > 0] = 0
    border = max(30, round(60 * resolution_scale))
    retained[:border] = retained[-border:] = 0
    retained[:, :border] = retained[:, -border:] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(retained, connectivity=8)
    linework = np.zeros_like(retained)
    for component in range(1, count):
        if max(stats[component, 2], stats[component, 3]) >= max(50, round(100 * resolution_scale)):
            linework[labels == component] = 1

    skeleton = skeletonize(linework > 0)
    paths = [
        np.asarray([(x, y) for y, x in path], dtype=float)
        for path in _walk_skeleton(skeleton)
        if len(path) >= 25
    ]
    fragment_count = len(paths)
    for gap, alignment in ((6, 0.82), (45, 0.93), (130, 0.965), (210, 0.985)):
        paths = _stitch(
            paths,
            max_gap=gap * resolution_scale,
            minimum_alignment=alignment,
        )
    minimum_length = max(75, round(150 * resolution_scale))
    paths = [path for path in paths if len(path) >= minimum_length]
    simplified = []
    for path in paths:
        contour = path.astype(np.float32).reshape(-1, 1, 2)
        points = cv2.approxPolyDP(contour, 2.0 * resolution_scale, False).reshape(-1, 2)
        if len(points) >= 2:
            simplified.append(points.tolist())

    return {
        "polylines": simplified,
        "isoline_mask": np.uint8(linework * 255),
        "profile_mask": np.uint8(profile_mask * 255),
        "fragment_count": fragment_count,
        "profile_segments": segment_count,
        "dash_links": dash_links,
        "source_resolution": [width, height],
        "resolution_scale": resolution_scale,
    }
