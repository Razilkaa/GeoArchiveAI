"""Trace long authored contours from archival map scans.

This is the production form of the original sheet-23 tracer.  It deliberately
works at source resolution: small connected components are text/ticks, long
straight components are profiles/frame, and the remaining skeleton is joined
through gaps only when endpoint tangents agree.
"""
from __future__ import annotations
import math


import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def _prune_spurs(skeleton: np.ndarray, max_length: int, rounds: int = 4) -> np.ndarray:
    """Remove short dead-end branches so contours stop fragmenting at them.

    Depth ticks, label strokes and skeletonization noise touch a contour and
    leave a stub. Every stub is a junction, and a junction splits the contour
    into separate walks, so a contour crossed ten times arrives as eleven
    fragments that later filters then discard piecewise. Cutting the stubs
    first keeps the contour a single path.
    """
    pruned = skeleton.copy()
    offsets = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    for _ in range(rounds):
        pixels = set(map(tuple, np.argwhere(pruned)))
        if not pixels:
            break

        def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
            y, x = point
            return [(y + dy, x + dx) for dy, dx in offsets if (y + dy, x + dx) in pixels]

        degree = {point: len(neighbours(point)) for point in pixels}
        removed: list[tuple[int, int]] = []
        for endpoint in [point for point, count in degree.items() if count == 1]:
            branch = [endpoint]
            previous, current = None, endpoint
            while len(branch) <= max_length:
                candidates = [item for item in neighbours(current) if item != previous]
                if len(candidates) != 1:
                    break
                previous, current = current, candidates[0]
                if degree[current] > 2:
                    break
                branch.append(current)
            # Only cut when the branch really ends in a junction: an isolated
            # short line is linework, not a stub.
            if len(branch) <= max_length and degree[current] > 2:
                removed.extend(branch)
        if not removed:
            break
        rows, columns = zip(*removed)
        pruned[np.asarray(rows), np.asarray(columns)] = False
    return pruned


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


def _arc_offset(line: np.ndarray, at_start: bool, target: np.ndarray, count: int = 28) -> float:
    """How far ``target`` sits off the arc this line end is drawing.

    Direction alone cannot tell a dashed contour continuing from the next
    contour running beside it: both point the same way. Their curvature does
    differ, and a neighbour sits off the arc by the contour spacing, so fitting
    a circle to the end of a stroke and measuring the target against it
    separates the two.
    """
    segment = line[:count] if at_start else line[-count:][::-1]
    if len(segment) < 6:
        return 0.0
    x, y = segment[:, 0], segment[:, 1]
    matrix = np.column_stack([x, y, np.ones(len(segment))])
    rhs = -(x**2 + y**2)
    try:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    center = np.array([-solution[0] / 2.0, -solution[1] / 2.0])
    squared = float(center[0] ** 2 + center[1] ** 2 - solution[2])
    span = float(np.linalg.norm(segment[0] - segment[-1]))
    if squared <= 0.0 or math.sqrt(squared) > span * 40.0:
        # Effectively straight: measure against the fitted line instead, so a
        # long flat stroke is not judged by a numerically unstable circle.
        direction = segment[0] - segment[-1]
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return 0.0
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]])
        return abs(float(normal @ (target - segment[0])))
    radius = math.sqrt(squared)
    return abs(float(np.linalg.norm(target - center)) - radius)


def _stitch(
    paths: list[np.ndarray],
    *,
    max_gap: float,
    minimum_alignment: float,
    arc_tolerance: float | None = None,
    arc_minimum_gap: float = 12.0,
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
            # Short joins continue a stroke that was merely nicked; it is the
            # long reaches that can land on the contour running alongside, so
            # only those are worth paying the arc test's false rejections for.
            if arc_tolerance is not None and distance > arc_minimum_gap:
                tolerance = max(arc_tolerance, distance * 0.40)
                if (
                    _arc_offset(first, start_a, point_b) > tolerance
                    or _arc_offset(second, start_b, point_a) > tolerance
                ):
                    continue
            joined_a = first[::-1] if start_a else first
            joined_b = second if start_b else second[::-1]
            lines[line_a] = np.vstack([joined_a, joined_b])
            lines[line_b] = None
            used.update((left_index, right_index))
            changed = True
    return [line for line in lines if line is not None]


def trace_source_geometry(
    gray: np.ndarray,
    *,
    component_span_factor: float = 80.0,
    profile_length_factor: float = 750.0,
    output_length_factor: float = 100.0,
) -> dict:
    ink = np.uint8(gray < 128)
    height, width = ink.shape
    resolution_scale = max(0.5, min(height, width) / 8_000.0)
    min_component_span = max(40, round(component_span_factor * resolution_scale))

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
        minLineLength=max(250, round(profile_length_factor * resolution_scale)),
        maxLineGap=max(6, round(12 * resolution_scale)),
    )
    segment_count = 0
    if segments is not None:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            cv2.line(profile_mask, (x1, y1), (x2, y2), 1, max(3, round(9 * resolution_scale)))
            segment_count += 1
    # Removing the whole corridor also cuts every contour that crosses it, and
    # restoring those crossings by stroke orientation was measured end to end:
    # it recovers mask ink but yields no better contour layer, because the
    # wobbly profile ink it keeps returns as decoration. The plain band stays.
    retained[profile_mask > 0] = 0
    border = max(30, round(60 * resolution_scale))
    retained[:border] = retained[-border:] = 0
    retained[:, :border] = retained[:, -border:] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(retained, connectivity=8)
    linework = np.zeros_like(retained)
    for component in range(1, count):
        if max(stats[component, 2], stats[component, 3]) >= max(50, round(100 * resolution_scale)):
            linework[labels == component] = 1

    skeleton = _prune_spurs(
        skeletonize(linework > 0), max_length=max(8, round(18 * resolution_scale))
    )
    # Short chains between two junctions are what connects a contour across a
    # crossing, so they must reach the stitcher rather than be filtered out by
    # length beforehand. The floor stays high enough to keep skeleton noise in
    # dense label areas out: below it, stray chains stitch into wandering paths
    # that the decoration filters then have to throw away wholesale.
    paths = [
        np.asarray([(x, y) for y, x in path], dtype=float)
        for path in _walk_skeleton(skeleton)
        if len(path) >= 25
    ]
    fragment_count = len(paths)
    # Merely widening the angle window to bridge the dashed Volga-Ural contours
    # was measured and rejected: it hopped between neighbouring contours. The
    # arc test below is what lets the window open, because a neighbour sits off
    # the arc even when it points the same way.
    for gap, alignment in ((6, 0.82), (45, 0.93), (130, 0.965), (210, 0.985)):
        paths = _stitch(
            paths,
            max_gap=gap * resolution_scale,
            minimum_alignment=alignment,
            arc_tolerance=14.0 * resolution_scale,
            arc_minimum_gap=60.0 * resolution_scale,
        )
    minimum_length = max(60, round(output_length_factor * resolution_scale))
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
