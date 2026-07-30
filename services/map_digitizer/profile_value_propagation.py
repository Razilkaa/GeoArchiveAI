"""Propagate contour values along seismic profile lines.

A seismic profile crosses the isoline bundle in depth order.  Anchors along a
profile are crossings of contours that already carry trusted values and OCR
depth marks sitting on the profile itself.  Between two anchors the contour
levels are a known finite set (multiples of the interval strictly inside the
anchor values), so when the number of unlabeled crossings matches, values
follow by pure counting; when the traced contours are fragmented, the
preliminary surface prediction picks which levels are present, still bounded
by the anchors and forced to stay monotone.  Every profile-contour crossing
with a value becomes an honest depth point.
"""
from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import nearest_points


def extract_profile_lines(
    profile_mask: np.ndarray,
    *,
    max_lines: int = 60,
) -> list[dict]:
    """Cluster Hough segments of the profile mask into straight profile lines.

    Returns lines in mask pixel coordinates: anchor point, unit direction and
    the parameter extent actually supported by mask ink (plus a small margin,
    because contours continue slightly past the drawn profile ends).
    """
    height, width = profile_mask.shape
    scale = min(1.0, 2200.0 / max(height, width))
    small = cv2.resize(
        profile_mask,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    small = np.uint8(small > 0) * 255
    segments = cv2.HoughLinesP(
        small,
        1,
        np.pi / 720,
        threshold=60,
        minLineLength=max(40, round(min(small.shape) * 0.04)),
        maxLineGap=max(10, round(min(small.shape) * 0.012)),
    )
    if segments is None:
        return []

    entries = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).astype(float):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1.0:
            continue
        entries.append(((x1, y1), (x2, y2), length))
    entries.sort(key=lambda item: -item[2])

    offset_limit = max(8.0, min(small.shape) * 0.006)

    def matches(cluster: dict, start, end, direction) -> bool:
        if abs(float(np.dot(direction, cluster["direction"]))) < 0.9993:
            return False
        for point in (start, end):
            delta = np.asarray(point, dtype=float) - cluster["anchor"]
            offset = abs(
                float(
                    cluster["direction"][0] * delta[1]
                    - cluster["direction"][1] * delta[0]
                )
            )
            if offset > offset_limit:
                return False
        return True

    def refit(cluster: dict) -> None:
        points = np.asarray(cluster["points"], dtype=float)
        mean = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - mean)
        cluster["anchor"] = mean
        cluster["direction"] = vectors[0]

    clusters: list[dict] = []
    for start, end, length in entries:
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        direction /= np.linalg.norm(direction)
        placed = False
        for cluster in clusters:
            if not matches(cluster, start, end, direction):
                continue
            cluster["points"].extend((start, end))
            cluster["weight"] += length
            refit(cluster)
            placed = True
            break
        if not placed:
            clusters.append(
                {
                    "anchor": np.asarray(start, dtype=float),
                    "direction": direction,
                    "points": [start, end],
                    "weight": length,
                }
            )

    # A long profile often arrives as several Hough clusters; join collinear
    # clusters whose fitted lines coincide.
    merged = True
    while merged:
        merged = False
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left, right = clusters[left_index], clusters[right_index]
                candidates = np.asarray(right["points"], dtype=float)
                deltas = candidates - left["anchor"]
                offsets = np.abs(
                    left["direction"][0] * deltas[:, 1]
                    - left["direction"][1] * deltas[:, 0]
                )
                if (
                    abs(float(np.dot(left["direction"], right["direction"]))) < 0.9993
                    or float(offsets.max()) > offset_limit * 1.5
                ):
                    continue
                left["points"].extend(right["points"])
                left["weight"] += right["weight"]
                refit(left)
                del clusters[right_index]
                merged = True
                break
            if merged:
                break

    lines = []
    minimum_weight = min(small.shape) * 0.05
    for cluster in sorted(clusters, key=lambda item: -item["weight"])[:max_lines]:
        if cluster["weight"] < minimum_weight:
            continue
        points = np.asarray(cluster["points"], dtype=float)
        mean = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - mean)
        direction = vectors[0]
        offsets = (points - mean) @ direction
        margin = min(small.shape) * 0.06
        lines.append(
            {
                "anchor": mean / scale,
                "direction": direction,
                "t_min": (float(offsets.min()) - margin) / scale,
                "t_max": (float(offsets.max()) + margin) / scale,
                "support_px": cluster["weight"] / scale,
            }
        )
    return lines


def _local_direction(geometry: LineString, point: Point) -> np.ndarray:
    distance = geometry.project(point)
    ahead = geometry.interpolate(min(geometry.length, distance + 12.0))
    behind = geometry.interpolate(max(0.0, distance - 12.0))
    vector = np.asarray([ahead.x - behind.x, ahead.y - behind.y])
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _crossing_events(
    profile: dict,
    geometries: dict[int, LineString],
    marks: np.ndarray | None,
    *,
    corridor: float = 30.0,
    mark_distance: float = 45.0,
    group_distance: float = 28.0,
) -> list[dict]:
    """Ordered anchor/crossing events along one profile line.

    Contour crossings closer than ``group_distance`` along the profile are one
    drafting-scale crossing of a fragmented authored contour and are grouped.
    OCR depth marks within ``mark_distance`` of the line become point events.
    """
    anchor = np.asarray(profile["anchor"], dtype=float)
    direction = np.asarray(profile["direction"], dtype=float)
    start = anchor + direction * profile["t_min"]
    end = anchor + direction * profile["t_max"]
    segment = LineString([tuple(start), tuple(end)])
    raw = []
    for line_id, geometry in geometries.items():
        points: list[Point] = []
        if segment.intersects(geometry):
            crossing = segment.intersection(geometry)
            if isinstance(crossing, Point):
                points = [crossing]
            elif isinstance(crossing, MultiPoint):
                points = list(crossing.geoms)
            else:
                points = [
                    geom
                    for geom in getattr(crossing, "geoms", [])
                    if isinstance(geom, Point)
                ]
        elif segment.distance(geometry) <= corridor:
            # Traced contours are broken exactly where the profile ink was
            # subtracted; a transverse approach into the corridor is still a
            # crossing of the authored contour.
            points = [nearest_points(segment, geometry)[1]]
        for point in points:
            tangent = _local_direction(geometry, point)
            transversality = abs(
                float(direction[0] * tangent[1] - direction[1] * tangent[0])
            )
            if transversality < 0.35:
                continue
            offset = float(np.dot(np.asarray([point.x, point.y]) - anchor, direction))
            if offset < profile["t_min"] or offset > profile["t_max"]:
                continue
            raw.append({"t": offset, "x": point.x, "y": point.y, "line_id": line_id})
    raw.sort(key=lambda item: item["t"])

    events: list[dict] = []
    for item in raw:
        if events and events[-1]["kind"] == "crossing":
            previous = events[-1]
            if item["line_id"] in previous["line_ids"]:
                continue
            if item["t"] - previous["t"] <= group_distance:
                previous["line_ids"].append(item["line_id"])
                continue
        events.append(
            {
                "kind": "crossing",
                "t": item["t"],
                "x": item["x"],
                "y": item["y"],
                "line_ids": [item["line_id"]],
            }
        )

    if marks is not None and len(marks):
        deltas = marks[:, :2] - anchor
        offsets = deltas @ direction
        distances = np.abs(direction[0] * deltas[:, 1] - direction[1] * deltas[:, 0])
        keep = (
            (distances <= mark_distance)
            & (offsets >= profile["t_min"])
            & (offsets <= profile["t_max"])
        )
        for x, y, value in zip(
            marks[keep, 0], marks[keep, 1], marks[keep, 2], strict=True
        ):
            events.append(
                {
                    "kind": "mark",
                    "t": float(np.dot(np.asarray([x, y]) - anchor, direction)),
                    "x": float(x),
                    "y": float(y),
                    "value_km": float(value),
                }
            )
        events.sort(key=lambda item: item["t"])
    return events


def _anchor_value(event: dict, known: dict[int, float]) -> float | None:
    if event["kind"] == "mark":
        return event["value_km"]
    values = {known[line_id] for line_id in event["line_ids"] if line_id in known}
    if len(values) == 1:
        return values.pop()
    return None


def _levels_between(low: float, high: float, interval: float) -> list[float]:
    first = math.floor(low / interval + 1e-6) + 1
    last = math.ceil(high / interval - 1e-6) - 1
    return [round(index * interval, 3) for index in range(first, last + 1)]


def _bracket_votes(
    events: list[dict],
    known: dict[int, float],
    interval: float,
    profile_id: int,
    *,
    max_span_km: float = 2.0,
) -> list[dict]:
    votes = []
    anchors = [
        (index, value)
        for index, event in enumerate(events)
        if (value := _anchor_value(event, known)) is not None
    ]
    for (left, value_left), (right, value_right) in zip(anchors, anchors[1:]):
        inner = [
            event
            for event in events[left + 1 : right]
            if event["kind"] == "crossing"
            and not any(line_id in known for line_id in event["line_ids"])
        ]
        if not inner:
            continue
        low, high = sorted((value_left, value_right))
        if high - low > max_span_km:
            continue
        levels = _levels_between(low, high, interval)
        if not levels:
            continue
        ascending = value_right > value_left
        ordered_levels = levels if ascending else levels[::-1]
        if len(inner) == len(levels):
            # Exact count match: values follow by pure counting.
            assigned = ordered_levels
        else:
            # Fragmented contours make the raw count unreliable; let the
            # preliminary surface pick which levels are present, still bounded
            # by the anchors and forced to stay monotone toward the far anchor.
            predictions = [event.get("prediction") for event in inner]
            if any(prediction is None for prediction in predictions):
                continue
            assigned = [
                min(levels, key=lambda level: abs(level - prediction))
                for prediction in predictions
            ]
            deltas = np.diff([value_left, *assigned, value_right])
            if ascending and np.any(deltas < -1e-9):
                continue
            if not ascending and np.any(deltas > 1e-9):
                continue
        for event, value in zip(inner, assigned, strict=True):
            for line_id in event["line_ids"]:
                votes.append(
                    {
                        "line_id": line_id,
                        "value_km": round(value, 3),
                        "profile_id": profile_id,
                    }
                )
    return votes


def propagate_profile_values(
    geometries: dict[int, LineString],
    seed_values: dict[int, float],
    profile_lines: list[dict],
    *,
    interval: float,
    predict=None,
    depth_marks: np.ndarray | None = None,
    corridor: float = 30.0,
    max_rounds: int = 6,
) -> dict:
    """Assign contour values by counting crossings along profiles.

    ``geometries`` maps trace id to its LineString (profile suspects excluded
    by the caller).  ``seed_values`` are trusted values in km (negative depth).
    ``depth_marks`` is an (N, 3) array of x, y, value_km (negative) OCR depth
    marks used as additional point anchors.  ``predict(x, y)`` optionally
    returns the preliminary-surface depth in km (negative) used when the pure
    crossing count does not close a bracket.  Propagated values become seeds
    for the next round until a fixed point.
    """
    crossings: dict[int, list[dict]] = {}
    mark_anchor_count = 0
    for profile_id, profile in enumerate(profile_lines):
        events = _crossing_events(
            profile, geometries, depth_marks, corridor=corridor
        )
        for event in events:
            if event["kind"] != "crossing":
                continue
            prediction = predict(event["x"], event["y"]) if predict else None
            event["prediction"] = (
                float(prediction)
                if prediction is not None and np.isfinite(prediction)
                else None
            )
        if sum(event["kind"] == "crossing" for event in events) >= 1 and len(events) >= 2:
            crossings[profile_id] = events
            mark_anchor_count += sum(event["kind"] == "mark" for event in events)

    known = dict(seed_values)
    propagated: dict[int, float] = {}
    conflicts: dict[int, list[float]] = {}
    rounds = 0
    for _ in range(max_rounds):
        rounds += 1
        votes: dict[int, Counter] = {}
        for profile_id, events in crossings.items():
            for vote in _bracket_votes(events, known, interval, profile_id):
                if vote["line_id"] in known:
                    continue
                votes.setdefault(vote["line_id"], Counter())[vote["value_km"]] += 1
        changed = False
        for line_id, counter in votes.items():
            value, count = counter.most_common(1)[0]
            dissent = sum(counter.values()) - count
            if dissent and count < 2 * dissent:
                conflicts[line_id] = sorted(counter)
                continue
            known[line_id] = value
            propagated[line_id] = value
            conflicts.pop(line_id, None)
            changed = True
        if not changed:
            break

    points = []
    for profile_id, events in crossings.items():
        for event in events:
            if event["kind"] != "crossing":
                continue
            for line_id in event["line_ids"]:
                value = known.get(line_id)
                if value is None:
                    continue
                points.append(
                    {
                        "x": round(event["x"], 1),
                        "y": round(event["y"], 1),
                        "value_km": value,
                        "profile_id": profile_id,
                        "line_id": line_id,
                        "source": "direct"
                        if line_id in seed_values
                        else "profile_order",
                    }
                )
    return {
        "profile_lines": len(profile_lines),
        "profiles_with_crossings": len(crossings),
        "crossing_events": sum(
            sum(event["kind"] == "crossing" for event in events)
            for events in crossings.values()
        ),
        "mark_anchors": mark_anchor_count,
        "seed_traces": len(seed_values),
        "propagated_values": propagated,
        "conflicts": {key: value for key, value in conflicts.items()},
        "rounds": rounds,
        "points": points,
    }
