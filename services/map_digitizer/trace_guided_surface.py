"""Snap traced source-map contours to an OCR-derived depth surface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from services.map_digitizer.assign_contour_values import assign_values, straightness
from services.map_digitizer.contour_cleanup import remove_unsupported_closed_contours
from services.map_digitizer.profile_value_propagation import (
    extract_profile_lines,
    propagate_profile_values,
)
from services.map_digitizer.depth_mark_surface import (
    extract_measurements,
    filter_depth_marks,
    image_scale,
    load_ocr_readings,
    merge_nearby_measurements,
)
from services.map_digitizer.export_cps3_grid import (
    build_harmonic_grid,
    contour_topology,
    extract_surface_contours,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    line = LineString(points)
    count = max(2, math.ceil(line.length / spacing) + 1)
    return np.asarray(
        [[point.x, point.y] for point in (line.interpolate(d) for d in np.linspace(0, line.length, count))]
    )


def assign_traced_values(
    polylines: list[np.ndarray],
    interpolator: RegularGridInterpolator,
    *,
    interval: float,
    sample_spacing: float = 20.0,
    minimum_coverage: float = 0.2,
    maximum_spread: float = 0.4,
    maximum_residual: float = 0.12,
    minimum_length: float = 0.0,
) -> list[dict]:
    assignments = []
    for index, points in enumerate(polylines):
        samples = densify(points, sample_spacing)
        predictions = interpolator(np.column_stack([samples[:, 1], samples[:, 0]]))
        valid = np.isfinite(predictions)
        coverage = float(np.mean(valid))
        length = float(LineString(points).length)
        curvature = straightness(points)
        profile_suspect = (
            (curvature < 0.01 and length > 100.0)
            or (curvature < 0.035 and length > 400.0)
        )
        valid_count = int(np.sum(valid))
        if valid_count >= 2:
            values = np.abs(predictions[valid])
            median = float(np.median(values))
            snapped = round(median / interval) * interval
            spread = float(np.percentile(values, 90) - np.percentile(values, 10))
            residual = float(np.median(np.abs(values - snapped)))
        else:
            snapped = spread = residual = None
        accepted = bool(
            not profile_suspect
            and length >= minimum_length
            and coverage >= minimum_coverage
            and valid_count >= 5
            and residual is not None
            and residual <= maximum_residual
            and spread <= maximum_spread
        )
        assignments.append(
            {
                "id": index,
                "accepted": accepted,
                "profile_suspect": profile_suspect,
                "coverage": round(coverage, 4),
                "length_px": round(length, 2),
                "value_km": -round(float(snapped), 3) if accepted else None,
                "inferred_value_km": (
                    -round(float(snapped), 3) if snapped is not None else None
                ),
                "spread_km": round(float(spread), 4) if spread is not None else None,
                "residual_km": round(float(residual), 4) if residual is not None else None,
                "geometry": LineString(points),
            }
        )
    return assignments


VALUE_SOURCE_RANK = {
    "direct_ocr": 3,
    "profile_order": 2,
    "neighbor_order": 1,
    "grid_backfill": 1,
    "surface_interpolation": 1,
    "unassigned": 0,
}


def _estimate_continuation_gap(
    rows: list[dict], *, lower: float, upper: float
) -> float:
    """Size the join gap from the sheet's own dashing.

    A dashed contour breaks into pieces separated by roughly one dash cycle;
    a solid one only breaks where ink was lost. The typical gap between a trace
    end and the nearest continuing trace end therefore measures how this sheet
    is drawn, so the join reaches exactly across its dashes and no further.
    """
    if len(rows) < 6:
        return lower
    endpoints = []
    owners = []
    for index, row in enumerate(rows):
        coords = np.asarray(row["geometry"].coords)
        endpoints.append(coords[0])
        endpoints.append(coords[-1])
        owners.extend((index, index))
    points = np.asarray(endpoints)
    tree = cKDTree(points)
    distances, neighbours = tree.query(points, k=4)
    gaps = []
    for position in range(len(points)):
        for rank in range(1, 4):
            other = int(neighbours[position][rank])
            if owners[other] != owners[position]:
                gaps.append(float(distances[position][rank]))
                break
    if not gaps:
        return lower
    return float(np.clip(np.median(gaps) * 1.5, lower, upper))


def unify_continuation_values(
    rows: list[dict], *, max_gap: float, minimum_alignment: float = 0.72
) -> dict:
    """Give one authored contour one level along its whole length.

    Fragments of the same contour are valued independently, so a contour that
    was traced in three pieces can come out carrying three different levels —
    visibly one line drawn in three colours. Geometry says they are one line,
    so the strongest evidence in the chain decides for all of it. Where two
    label-backed fragments in a chain disagree, the chain is left alone: that
    is a real conflict to review, not something to average away.

    The continuation test matches ``dissolve_by_level`` on purpose. Anything
    that step would merge into one isoline is one contour, so judging it by a
    stricter rule here left exactly those pieces disagreeing and unmerged.
    """
    if len(rows) < 2:
        return {"groups": 0, "harmonized": 0, "conflicts": 0}

    endpoints = []
    for index, row in enumerate(rows):
        coordinates = np.asarray(row["geometry"].coords)
        endpoints.append((coordinates[0], index, True, coordinates))
        endpoints.append((coordinates[-1], index, False, coordinates))

    def tangent(points: np.ndarray, at_start: bool, count: int = 12) -> np.ndarray:
        segment = points[:count] if at_start else points[-count:][::-1]
        vector = segment[0] - segment[-1]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    tree = cKDTree(np.asarray([item[0] for item in endpoints]))
    for left, right in tree.query_pairs(max_gap):
        point_a, index_a, start_a, coords_a = endpoints[left]
        point_b, index_b, start_b, coords_b = endpoints[right]
        if index_a == index_b:
            continue
        tangent_a = tangent(coords_a, start_a)
        tangent_b = tangent(coords_b, start_b)
        bridge = point_b - point_a
        distance = float(np.linalg.norm(bridge))
        continuation = float(np.dot(tangent_a, -tangent_b))
        if distance > 2.0:
            direction = bridge / distance
            continuation = min(
                continuation,
                float(np.dot(tangent_a, direction)),
                float(np.dot(-tangent_b, direction)),
            )
        if continuation < minimum_alignment:
            continue
        root_a, root_b = find(index_a), find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)

    harmonized = 0
    conflicts = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        ranked = [
            (VALUE_SOURCE_RANK.get(rows[index]["value_source"], 0), index)
            for index in members
            if rows[index]["value_km"] is not None
        ]
        if not ranked:
            continue
        best_rank = max(rank for rank, _ in ranked)
        leaders = [index for rank, index in ranked if rank == best_rank]
        distinct = {rows[index]["value_km"] for index in leaders}
        if len(distinct) > 1:
            # Only evidence that read the sheet — a label, or an ordering along
            # a profile — can veto. Two inferred levels disagreeing is not a
            # finding about the map, so the longer geometry simply wins.
            if best_rank >= 2:
                conflicts += 1
                continue
            weights: dict[float, float] = {}
            for index in leaders:
                key = rows[index]["value_km"]
                weights[key] = weights.get(key, 0.0) + rows[index]["geometry"].length
            value = max(weights, key=weights.get)
        else:
            value = rows[leaders[0]]["value_km"]
        for index in members:
            if rows[index]["value_km"] == value:
                continue
            rows[index]["value_km"] = value
            rows[index]["value_m"] = value * 1000.0
            rows[index]["value_source"] = "continuation_consensus"
            harmonized += 1
    return {
        "groups": sum(1 for members in groups.values() if len(members) > 1),
        "harmonized": harmonized,
        "conflicts": conflicts,
    }


def dissolve_by_level(
    items: list[dict], *, max_gap: float = 160.0, minimum_alignment: float = 0.72
) -> list[dict]:
    """Join same-level trace fragments into continuous authored isolines.

    Fragments sharing one contour value belong to the same authored contour
    when their endpoints continue each other, so joining is safe there even
    across label-sized gaps that generic stitching must refuse.
    """
    chains: list[dict] = []
    for item in items:
        if item.get("value_km") is None:
            continue
        chains.append(
            {
                "value_km": item["value_km"],
                "points": np.asarray(item["geometry"].coords, dtype=float),
                "members": [int(item["trace_id"])],
            }
        )

    def tangent(points: np.ndarray, at_start: bool) -> np.ndarray:
        count = min(10, len(points))
        segment = points[:count] if at_start else points[-count:][::-1]
        vector = segment[0] - segment[-1]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    changed = True
    while changed:
        changed = False
        best = None
        for left_index in range(len(chains)):
            for right_index in range(left_index + 1, len(chains)):
                left, right = chains[left_index], chains[right_index]
                if left["value_km"] != right["value_km"]:
                    continue
                for start_a in (True, False):
                    for start_b in (True, False):
                        point_a = left["points"][0 if start_a else -1]
                        point_b = right["points"][0 if start_b else -1]
                        bridge = point_b - point_a
                        distance = float(np.linalg.norm(bridge))
                        if distance > max_gap:
                            continue
                        tangent_a = tangent(left["points"], start_a)
                        tangent_b = tangent(right["points"], start_b)
                        continuation = float(np.dot(tangent_a, -tangent_b))
                        if distance > 2.0:
                            direction = bridge / distance
                            continuation = min(
                                continuation,
                                float(np.dot(tangent_a, direction)),
                                float(np.dot(-tangent_b, direction)),
                            )
                        if continuation < minimum_alignment:
                            continue
                        score = distance * (2.0 - continuation)
                        if best is None or score < best[0]:
                            best = (score, left_index, right_index, start_a, start_b)
        if best is not None:
            _, left_index, right_index, start_a, start_b = best
            left, right = chains[left_index], chains[right_index]
            first = left["points"][::-1] if start_a else left["points"]
            second = right["points"] if start_b else right["points"][::-1]
            left["points"] = np.vstack([first, second])
            left["members"].extend(right["members"])
            del chains[right_index]
            changed = True
    return chains


def dissolve_and_resolve(rows: list[dict], *, max_gap: float) -> tuple[list[dict], int]:
    """Join same-level fragments into isolines and clear crossings between levels.

    Wraps the two steps that turn valued traces into published isolines so they
    can run twice: once to build the surface, once after fragments left over
    from that build have inherited a level from it.
    """
    dissolved = dissolve_by_level(rows, max_gap=max_gap)
    crossings = 0
    while True:
        chains = [LineString(chain["points"]) for chain in dissolved]
        tree = shapely.STRtree(chains)
        conflict = None
        for index, chain in enumerate(chains):
            for other in tree.query(chain):
                other = int(other)
                if other <= index:
                    continue
                if dissolved[index]["value_km"] == dissolved[other]["value_km"]:
                    continue
                if chain.crosses(chains[other]):
                    conflict = (index, other)
                    break
            if conflict:
                break
        if conflict is None:
            break
        loser = max(
            conflict,
            key=lambda index: (
                -len(dissolved[index]["members"]),
                -chains[index].length,
            ),
        )
        del dissolved[loser]
        crossings += 1
    return dissolved, crossings


def backfill_from_grid(
    rows: list[dict],
    interpolator: "RegularGridInterpolator",
    *,
    interval: float,
    residual_limit: float,
    spread_limit: float,
    source: str,
) -> int:
    """Give still-unlevelled fragments the level of the surface beneath them.

    Runs on the rebuilt surface, which honours whole assembled isolines, so a
    fragment that hugs one of its level lines inherits that exact level — the
    value of the isoline it abuts. A fragment cutting across levels fails the
    spread test and stays unvalued rather than being guessed.
    """
    filled = 0
    for row in rows:
        if row["value_km"] is not None:
            continue
        samples = densify(np.asarray(row["geometry"].coords), 20.0)
        sampled = (
            interpolator(np.column_stack([samples[:, 1], samples[:, 0]])) / 1000.0
        )
        valid = sampled[np.isfinite(sampled)]
        if len(valid) < 5:
            continue
        snapped = round(float(np.median(valid)) / interval) * interval
        residual = float(np.median(np.abs(valid - snapped)))
        spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
        if residual > interval * residual_limit or spread > interval * spread_limit:
            continue
        row["value_km"] = round(snapped, 3)
        row["value_m"] = row["value_km"] * 1000.0
        row["value_source"] = source
        filled += 1
    return filled


def reject_mark_outliers(
    assignments: list[dict],
    depth_marks: np.ndarray,
    *,
    interval: float,
    radius: float,
) -> list[int]:
    """Demote accepted traces whose level contradicts nearby profile depth marks.

    The authored isolines were drawn from the depth numbers posted along the
    seismic profiles, so a trace whose assigned level disagrees with the marks
    right next to it carries a misread value, not real relief.
    """
    tree = cKDTree(depth_marks[:, :2])
    demote_at = max(0.3, 3.0 * interval)
    null_at = max(0.45, 4.5 * interval)
    removed: list[int] = []
    for item in assignments:
        if not item["accepted"]:
            continue
        samples = densify(np.asarray(item["geometry"].coords), 40.0)
        indices = sorted(
            {index for hits in tree.query_ball_point(samples, r=radius) for index in hits}
        )
        if len(indices) < 4:
            continue
        marks = depth_marks[indices]
        distances = np.min(
            np.linalg.norm(marks[:, None, :2] - samples[None, :, :], axis=2), axis=1
        )
        nearest = np.argsort(distances)[:10]
        expected = float(np.median(marks[nearest, 2]))
        deviation = abs(float(item["value_km"]) - expected)
        if deviation <= demote_at:
            continue
        item["accepted"] = False
        item["rejection_reason"] = "depth_mark_outlier"
        item["expected_value_km"] = round(expected, 3)
        if deviation > null_at:
            item["value_km"] = None
            item["inferred_value_km"] = None
        removed.append(int(item["id"]))
    return removed


def rehabilitate_profile_suspects(
    assignments: list[dict],
    *,
    neighbourhood: float,
    tolerance_deg: float = 25.0,
    minimum_neighbours: int = 3,
) -> list[int]:
    """Recover straight isoline segments wrongly written off as profiles.

    Straightness alone cannot separate the two: an isoline on a gentle flank
    runs straight for a long way. What separates them is bearing. Isolines lie
    parallel to their neighbours, while a profile cuts across the field. So a
    straight trace that follows the local bearing of the surrounding contours
    is contour and returns to the layer; one that cuts across stays rejected.
    """
    reference = [
        item
        for item in assignments
        if not item["profile_suspect"] and item["length_px"] > 0.0
    ]
    suspects = [item for item in assignments if item["profile_suspect"]]
    if len(reference) < minimum_neighbours or not suspects:
        return []

    reference_centers = np.asarray(
        [item["geometry"].centroid.coords[0] for item in reference]
    )
    reference_headings = np.asarray(
        [_local_heading(np.asarray(item["geometry"].coords)) for item in reference]
    )
    tree = cKDTree(reference_centers)
    tolerance = math.radians(tolerance_deg)
    recovered: list[int] = []
    for item in suspects:
        center = np.asarray(item["geometry"].centroid.coords[0])
        neighbours = tree.query_ball_point(center, neighbourhood)
        if len(neighbours) < minimum_neighbours:
            continue
        heading = _local_heading(np.asarray(item["geometry"].coords))
        deltas = np.abs(reference_headings[neighbours] - heading)
        deltas = np.minimum(deltas, math.pi - deltas)
        # Half of the surrounding contours agreeing is enough: near a fold the
        # neighbourhood legitimately holds contours of two different bearings.
        if float(np.median(deltas)) > tolerance:
            continue
        item["profile_suspect"] = False
        item["rehabilitated"] = True
        recovered.append(int(item["id"]))
    return recovered


def _local_heading(points: np.ndarray) -> float:
    """Dominant direction of a polyline as an angle in [0, pi)."""
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-6
    if not np.any(keep):
        return 0.0
    segments, lengths = segments[keep], lengths[keep]
    doubled = 2.0 * np.arctan2(segments[:, 1], segments[:, 0])
    mean = np.arctan2(
        float(np.sum(np.sin(doubled) * lengths)),
        float(np.sum(np.cos(doubled) * lengths)),
    )
    return (mean / 2.0) % math.pi


def reject_profile_tick_combs(
    rows: list[dict],
    carriers: list[dict],
    *,
    maximum_length: float,
    maximum_offset: float,
    minimum_angle: float = 55.0,
    comb_radius: float,
) -> list[dict]:
    """Drop the tick marks drawn along seismic profiles.

    A profile is annotated with short strokes set across it at a regular
    spacing. Each stroke alone looks like a contour fragment, but the pattern
    gives them away: they are short, they sit square to their carrier line, and
    they come in combs. A contour that genuinely crosses a profile is a lone
    crossing, so requiring several neighbours on the same carrier keeps it.
    """
    if not carriers:
        return []

    candidates: dict[int, list[int]] = {}
    details: dict[int, np.ndarray] = {}
    for index, row in enumerate(rows):
        points = np.asarray(row["geometry"].coords)
        if row["geometry"].length > maximum_length:
            continue
        centre = points.mean(axis=0)
        heading = math.degrees(_local_heading(points)) % 180.0
        nearest = None
        for carrier in carriers:
            offset = abs(
                float(carrier["normal"] @ (centre - carrier["anchor"]))
            )
            if nearest is None or offset < nearest[0]:
                nearest = (offset, carrier["id"], carrier["angle"])
        if nearest is None or nearest[0] > maximum_offset:
            continue
        deviation = abs(heading - nearest[2]) % 180.0
        deviation = min(deviation, 180.0 - deviation)
        if deviation < minimum_angle:
            continue
        candidates.setdefault(nearest[1], []).append(index)
        details[index] = centre

    removed: list[dict] = []
    doomed: set[int] = set()
    for members in candidates.values():
        if len(members) < 3:
            continue
        centres = np.asarray([details[index] for index in members])
        spacing = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
        for position, index in enumerate(members):
            neighbours = np.sort(spacing[position])[1:4]
            if int(np.sum(neighbours < comb_radius)) >= 2:
                doomed.add(index)
    for index in sorted(doomed, reverse=True):
        row = rows.pop(index)
        row["rejection_reason"] = "profile_tick_comb"
        removed.append(row)
    return removed


def prune_crossing_traces(rows: list[dict], *, neighbourhood: float) -> list[dict]:
    """Remove traces that cut across the isoline field instead of following it.

    Authored isolines never intersect, so every crossing means at least one of
    the two lines is not an isoline. The offender is the one that crosses many
    neighbours and runs against the local trend of the surrounding linework:
    a fault trace, a profile remnant or a stitched-together decoration. Removal
    is greedy and rechecked, so a genuine isoline crossed once by an artifact
    survives while the artifact goes.
    """
    removed: list[dict] = []
    live = list(rows)
    while True:
        geometries = [row["geometry"] for row in live]
        if len(geometries) < 2:
            break
        tree = shapely.STRtree(geometries)
        crossings = [0] * len(live)
        for index, geometry in enumerate(geometries):
            for other in tree.query(geometry):
                if int(other) != index and geometry.crosses(geometries[int(other)]):
                    crossings[index] += 1
        if not any(crossings):
            break

        centers = np.asarray([geometry.centroid.coords[0] for geometry in geometries])
        headings = np.asarray(
            [_local_heading(np.asarray(geometry.coords)) for geometry in geometries]
        )
        neighbour_tree = cKDTree(centers)
        scores = []
        for index, count in enumerate(crossings):
            if not count:
                continue
            neighbours = [
                other
                for other in neighbour_tree.query_ball_point(centers[index], neighbourhood)
                if other != index
            ]
            if neighbours:
                deltas = np.abs(headings[neighbours] - headings[index])
                disagreement = float(np.mean(np.minimum(deltas, math.pi - deltas)))
            else:
                disagreement = 0.0
            scores.append((count, disagreement, geometries[index].length, index))
        if not scores:
            break
        _, _, _, worst = max(scores, key=lambda item: (item[0], item[1], item[2]))
        row = live.pop(worst)
        row["rejection_reason"] = "crosses_isoline_field"
        removed.append(row)
    rows[:] = live
    return removed


def prune_crossing_constraints(assignments: list[dict]) -> list[int]:
    """Reject the weaker member of every different-level crossing pair."""
    removed: list[int] = []
    while True:
        accepted = [item for item in assignments if item["accepted"]]
        conflict = None
        for left_index, left in enumerate(accepted):
            for right in accepted[left_index + 1 :]:
                if left["value_km"] != right["value_km"] and left["geometry"].crosses(right["geometry"]):
                    conflict = (left, right)
                    break
            if conflict:
                break
        if not conflict:
            return removed
        left, right = conflict
        def weakness(item: dict) -> tuple[float, float, float]:
            return (
                0.0
                if item.get("direct_label")
                else 0.5
                if item.get("profile_order")
                else 1.0,
                float(item.get("residual_km") or 1.0) + float(item.get("spread_km") or 1.0),
                -float(item.get("coverage") or 0.0),
            )
        loser = max((left, right), key=weakness)
        loser["accepted"] = False
        loser["rejection_reason"] = "crosses_different_level"
        removed.append(int(loser["id"]))


def run(
    traces_path: Path,
    point_surface_path: Path,
    output_dir: Path,
    *,
    interval: float,
    trace_image_path: Path | None = None,
    label_paths: list[Path] | None = None,
    long_contour_fraction: float = 0.35,
) -> dict:
    payload = json.loads(traces_path.read_text(encoding="utf-8"))
    raw_polylines = [np.asarray(points, dtype=float) for points in payload["polylines_xy"]]
    surface = np.load(point_surface_path)
    x, y, z = surface["x"], surface["y"], surface["z"]
    target_size = (float(x[-1] + 1), float(y[-1] + 1))

    source_path = trace_image_path or Path(payload.get("source", ""))
    with Image.open(source_path) as trace_image:
        trace_size = trace_image.size
    scale = np.asarray([target_size[0] / trace_size[0], target_size[1] / trace_size[1]])
    polylines = [points * scale for points in raw_polylines]
    interpolator = RegularGridInterpolator(
        (y, x), np.abs(z), bounds_error=False, fill_value=np.nan
    )
    minimum_trace_length = max(60.0, min(target_size) * 0.015)
    assignments = assign_traced_values(
        polylines,
        interpolator,
        interval=interval,
        minimum_length=minimum_trace_length,
    )
    finite_surface = np.abs(z[np.isfinite(z)])
    direct_low = -float(finite_surface.max()) - interval * 2
    direct_high = -float(finite_surface.min()) + interval * 2
    direct_labels: dict[int, float] = {}
    for label_path in label_paths or []:
        scale_x, scale_y, ocr_width, ocr_height = image_scale(
            label_path, (round(target_size[0]), round(target_size[1]))
        )
        readings = load_ocr_readings(label_path, (ocr_width, ocr_height))
        if abs(scale_x - 1.0) > 1e-4 or abs(scale_y - 1.0) > 1e-4:
            for reading in readings:
                reading["quad"] = [
                    [float(px) * scale_x, float(py) * scale_y]
                    for px, py in reading.get("quad") or []
                ]
        direct = assign_values(
            polylines, readings, interval=interval, max_label_distance=110.0
        )
        for index in direct["confident"]:
            value = float(direct["values"][index])
            if direct_low <= value <= direct_high:
                direct_labels[index] = value
    # Vetting labels against the depth-mark surface was measured and rejected:
    # it discards genuine labels near faults, where the smoothed surface
    # legitimately departs from the authored contour by several intervals.
    # A signed label outranks the surface, which is why it seeds it.
    for item in assignments:
        if item["id"] in direct_labels and not item["profile_suspect"]:
            item["value_km"] = direct_labels[item["id"]]
            item["inferred_value_km"] = direct_labels[item["id"]]
            item["accepted"] = True
            item["direct_label"] = True
        else:
            item["direct_label"] = False
        item["profile_order"] = False

    # Counting crossings along seismic profiles is stronger evidence than the
    # OCR point surface: it overrides surface-sampled levels, never direct ones.
    propagation = {
        "profile_lines": 0,
        "profiles_with_crossings": 0,
        "crossing_events": 0,
        "mark_anchors": 0,
        "seed_traces": len(direct_labels),
        "propagated_values": {},
        "conflicts": {},
        "rounds": 0,
        "points": [],
    }
    depth_marks = None
    mark_rows = []
    for label_path in label_paths or []:
        scale_x, scale_y, ocr_width, ocr_height = image_scale(
            label_path, (round(target_size[0]), round(target_size[1]))
        )
        readings = load_ocr_readings(label_path, (ocr_width, ocr_height))
        source_marks, _ = extract_measurements(
            readings, scale_x=scale_x, scale_y=scale_y, contour_interval=interval
        )
        if len(source_marks):
            mark_rows.append(source_marks)
    if mark_rows:
        stacked = filter_depth_marks(np.vstack(mark_rows))
        stacked = merge_nearby_measurements(
            stacked, max(8.0, min(target_size) * 0.003)
        )
        low_magnitude = float(finite_surface.min()) - interval * 2
        high_magnitude = float(finite_surface.max()) + interval * 2
        stacked = stacked[
            (stacked[:, 2] >= low_magnitude) & (stacked[:, 2] <= high_magnitude)
        ]
        if len(stacked):
            depth_marks = np.column_stack(
                [stacked[:, 0], stacked[:, 1], -np.abs(stacked[:, 2])]
            )
    profile_mask_path = traces_path.with_name("profile_mask.png")
    if profile_mask_path.exists() and (direct_labels or depth_marks is not None):
        mask = cv2.imdecode(
            np.fromfile(str(profile_mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if mask is not None:
            mask_scale = np.asarray(
                [target_size[0] / mask.shape[1], target_size[1] / mask.shape[0]]
            )
            profile_lines = extract_profile_lines(mask)
            for line in profile_lines:
                direction = line["direction"] * mask_scale
                stretch = float(np.linalg.norm(direction))
                line["anchor"] = line["anchor"] * mask_scale
                line["direction"] = direction / stretch
                line["t_min"] *= stretch
                line["t_max"] *= stretch
            geometries = {
                item["id"]: item["geometry"]
                for item in assignments
                if not item["profile_suspect"]
                and item["length_px"] >= minimum_trace_length * 0.5
            }
            def predict_depth_km(x: float, y: float) -> float:
                return -float(interpolator([[y, x]])[0])

            propagation = propagate_profile_values(
                geometries,
                {index: float(value) for index, value in direct_labels.items()},
                profile_lines,
                interval=interval,
                predict=predict_depth_km,
                depth_marks=depth_marks,
                corridor=max(20.0, min(target_size) * 0.004),
            )
            for item in assignments:
                value = propagation["propagated_values"].get(item["id"])
                if value is None or item["direct_label"]:
                    continue
                item["value_km"] = value
                item["inferred_value_km"] = value
                item["accepted"] = True
                item["profile_order"] = True
    mark_outlier_rejections: list[int] = []
    if depth_marks is not None and len(depth_marks) >= 20:
        mark_outlier_rejections = reject_mark_outliers(
            assignments,
            depth_marks,
            interval=interval,
            radius=max(90.0, min(target_size) * 0.015),
        )
    crossing_rejections = prune_crossing_constraints(assignments)
    accepted = [item for item in assignments if item["accepted"]]
    if len(accepted) < 3:
        raise ValueError(f"Only {len(accepted)} traced contours passed QC")

    # Straightness alone over-rejects: run the bearing test before deciding what
    # reaches the published layer. Recovered traces stay out of the grid
    # constraints, which were fixed above, so a mistake here cannot bend the
    # surface, only add linework.
    rehabilitated = rehabilitate_profile_suspects(
        assignments, neighbourhood=max(300.0, min(target_size) * 0.06)
    )

    # The traced geometry is the digitization result.  Grid acceptance is a
    # stricter decision used only to choose interpolation constraints; it must
    # not make the remaining source linework disappear from user exports.
    preserved_rows = []
    for item in assignments:
        if item["profile_suspect"] or item["length_px"] < minimum_trace_length * 0.5:
            continue
        value_km = item.get("value_km")
        if value_km is None:
            value_km = item.get("inferred_value_km")
        preserved_rows.append(
            {
                "trace_id": item["id"],
                "value_km": value_km,
                "value_m": value_km * 1000.0 if value_km is not None else None,
                "value_source": (
                    "direct_ocr"
                    if item.get("direct_label")
                    else "profile_order"
                    if item.get("profile_order")
                    else "surface_interpolation"
                    if value_km is not None
                    else "unassigned"
                ),
                "accepted_for_grid": bool(item["accepted"]),
                "coverage": item["coverage"],
                "spread_km": item["spread_km"],
                "residual_km": item["residual_km"],
                "geometry": item["geometry"],
            }
        )

    # Tick marks along the profiles survive tracing as short strokes set across
    # their carrier. They are decoration, so they leave before anything is
    # published or constrained.
    tick_carriers = []
    if profile_mask_path.exists():
        carrier_mask = cv2.imdecode(
            np.fromfile(str(profile_mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if carrier_mask is not None:
            carrier_scale = np.asarray(
                [
                    target_size[0] / carrier_mask.shape[1],
                    target_size[1] / carrier_mask.shape[0],
                ]
            )
            for line in extract_profile_lines(carrier_mask):
                direction = line["direction"] * carrier_scale
                stretch = float(np.linalg.norm(direction))
                if stretch == 0.0:
                    continue
                direction = direction / stretch
                tick_carriers.append(
                    {
                        "id": len(tick_carriers),
                        "anchor": line["anchor"] * carrier_scale,
                        "normal": np.array([-direction[1], direction[0]]),
                        "angle": math.degrees(
                            math.atan2(direction[1], direction[0])
                        )
                        % 180.0,
                    }
                )
    tick_traces = reject_profile_tick_combs(
        preserved_rows,
        tick_carriers,
        maximum_length=max(250.0, min(target_size) * 0.05),
        maximum_offset=max(40.0, min(target_size) * 0.007),
        comb_radius=max(600.0, min(target_size) * 0.09),
    )

    # Isolines do not intersect, so a trace that cuts across several of them is
    # decoration or a profile remnant. Dropping it here keeps it out of both the
    # published layer and the grid constraints.
    crossing_traces = prune_crossing_traces(
        preserved_rows, neighbourhood=max(200.0, min(target_size) * 0.05)
    )
    crossing_trace_ids = {int(row["trace_id"]) for row in crossing_traces}
    if crossing_trace_ids:
        for item in assignments:
            if item["id"] in crossing_trace_ids:
                item["accepted"] = False
                item["rejection_reason"] = "crosses_isoline_field"
        accepted = [item for item in assignments if item["accepted"]]
        if len(accepted) < 3:
            raise ValueError(f"Only {len(accepted)} traced contours passed QC")

    source_rows = [
        {
            "trace_id": item["id"],
            "value_km": item["value_km"],
            "coverage": item["coverage"],
            "spread_km": item["spread_km"],
            "residual_km": item["residual_km"],
            "direct_label": item["direct_label"],
            "geometry": item["geometry"],
        }
        for item in accepted
    ]
    contours = gpd.GeoDataFrame(source_rows, geometry="geometry", crs="EPSG:3857")
    support = gpd.GeoDataFrame(
        [
            {"value_km": 0.0, "geometry": item["geometry"]}
            for item in assignments
            if not item["profile_suspect"]
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    # Depth numbers posted along the profiles are the map author's own source
    # data, so they constrain the grid directly, at a weight below the traced
    # contours: contours keep the shape, marks pin the absolute level between
    # valued traces and suppress bullseyes around sparsely valued areas.
    # Short accepted traces are the ones most likely to be decoration that
    # slipped through, and each carries a level over only a few pixels, so it
    # buys almost no shape while it can bend the surface where it is wrong.
    # The published layer keeps everything; only the grid gets the long ones.
    grid_contours = contours
    constraint_length_floor = 0.0
    if len(contours) >= 12:
        lengths = np.asarray([geometry.length for geometry in contours.geometry])
        constraint_length_floor = float(np.mean(lengths) * long_contour_fraction)
        selected = contours[contours.geometry.length >= constraint_length_floor]
        # Never starve the solver: fall back if the floor leaves too little.
        if len(selected) >= max(8, len(contours) // 5):
            grid_contours = selected
    grid_constraints = grid_contours.assign(constraint_weight=25.0)
    if depth_marks is not None and len(depth_marks):
        mark_constraints = gpd.GeoDataFrame(
            {
                "value_km": depth_marks[:, 2],
                "constraint_weight": 6.0,
            },
            geometry=[Point(mark[0], mark[1]) for mark in depth_marks],
            crs="EPSG:3857",
        )
        grid_constraints = gpd.GeoDataFrame(
            pd.concat([grid_constraints, mark_constraints], ignore_index=True),
            geometry="geometry",
            crs="EPSG:3857",
        )
    grid, grid_quality = build_harmonic_grid(
        grid_constraints,
        cell_size=max(15.0, min(target_size) / 250.0),
        blanking_distance=min(target_size) / 12.0,
        constraint_weight=25.0,
        support_contours=support,
    )
    # Backfill levels the label and profile passes could not reach.  The final
    # grid already honours every valued contour and the profile depth marks, so
    # a trace that hugs one of its level lines can safely inherit that level;
    # traces that cut across grid levels stay unvalued rather than guessed.
    grid_interpolator = RegularGridInterpolator(
        (grid.y, grid.x), grid.z, bounds_error=False, fill_value=np.nan
    )
    grid_backfilled = 0
    for row in preserved_rows:
        if row["value_km"] is not None:
            continue
        samples = densify(np.asarray(row["geometry"].coords), 20.0)
        sampled = (
            grid_interpolator(np.column_stack([samples[:, 1], samples[:, 0]]))
            / 1000.0
        )
        valid = sampled[np.isfinite(sampled)]
        if len(valid) < 5:
            continue
        snapped = round(float(np.median(valid)) / interval) * interval
        residual = float(np.median(np.abs(valid - snapped)))
        spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
        if residual > interval * 0.35 or spread > interval * 0.9:
            continue
        row["value_km"] = round(snapped, 3)
        row["value_m"] = row["value_km"] * 1000.0
        row["value_source"] = "grid_backfill"
        grid_backfilled += 1

    # Where the smoothed grid drifts in absolute level (across fault zones),
    # the local level step between adjacent traces is still right, so packets
    # of parallel isolines inherit levels one trace at a time from valued
    # neighbours plus the grid's local difference.
    neighbor_valued = 0
    neighbor_spacing = max(150.0, min(target_size) * 0.03)
    for _ in range(6):
        valued_rows = [row for row in preserved_rows if row["value_km"] is not None]
        pending = [row for row in preserved_rows if row["value_km"] is None]
        if not pending or not valued_rows:
            break
        anchor_points_list = []
        anchor_values_list = []
        for row in valued_rows:
            anchor_samples = densify(np.asarray(row["geometry"].coords), 30.0)
            anchor_points_list.append(anchor_samples)
            anchor_values_list.extend([float(row["value_km"])] * len(anchor_samples))
        anchor_points = np.vstack(anchor_points_list)
        anchor_values = np.asarray(anchor_values_list)
        anchor_grid = grid_interpolator(anchor_points[:, [1, 0]]) / 1000.0
        anchor_tree = cKDTree(anchor_points)
        progress = False
        for row in pending:
            samples = densify(np.asarray(row["geometry"].coords), 30.0)
            grid_at = grid_interpolator(samples[:, [1, 0]]) / 1000.0
            distances, nearest = anchor_tree.query(samples, k=1)
            usable = np.isfinite(grid_at) & np.isfinite(anchor_grid[nearest])
            if int(np.sum(usable)) < 5:
                continue
            if float(np.median(distances[usable])) > neighbor_spacing:
                continue
            estimates = anchor_values[nearest[usable]] + (
                grid_at[usable] - anchor_grid[nearest[usable]]
            )
            snapped = round(float(np.median(estimates)) / interval) * interval
            residual = float(np.median(np.abs(estimates - snapped)))
            spread = float(np.percentile(estimates, 90) - np.percentile(estimates, 10))
            if residual > interval * 0.35 or spread > interval * 0.8:
                continue
            row["value_km"] = round(snapped, 3)
            row["value_m"] = row["value_km"] * 1000.0
            row["value_source"] = "neighbor_order"
            neighbor_valued += 1
            progress = True
        if not progress:
            break

    reconstructed = extract_surface_contours(grid, interval=interval * 1000.0)
    grid_spacing = float(np.median(np.diff(grid.x))) if len(grid.x) > 1 else 1.0
    reconstructed, contour_cleanup = remove_unsupported_closed_contours(
        reconstructed,
        contours,
        interval_m=interval * 1000.0,
        support_distance=grid_spacing * 4.0,
    )
    topology = contour_topology(reconstructed)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_points_path = output_dir / "profile_points.csv"
    with profile_points_path.open("w", encoding="utf-8-sig") as handle:
        handle.write("x_px,y_px,depth_km,profile_id,line_id,source\n")
        for point in propagation["points"]:
            handle.write(
                f"{point['x'] / scale[0]:.1f},{point['y'] / scale[1]:.1f},"
                f"{point['value_km']},{point['profile_id']},"
                f"{point['line_id']},{point['source']}\n"
            )
    # One contour must carry one level before anything is written or joined:
    # the per-trace layer and the isoline layer are published side by side, so
    # harmonizing after the first is written leaves the two disagreeing. The gap
    # must reach across a dash cycle, but only as far as the traces themselves
    # are broken — a fixed fraction of the sheet is too wide on a dense sheet
    # (it pairs contours that merely run close) and too narrow on a dashed one.
    # The median end-to-end gap between distinct traces measures the real dash
    # cycle of this sheet; the same value feeds the dissolve below so the two
    # steps never disagree about what is one contour.
    continuation_gap = _estimate_continuation_gap(
        preserved_rows,
        lower=max(120.0, min(target_size) * 0.015),
        upper=min(target_size) * 0.04,
    )
    continuation = unify_continuation_values(
        preserved_rows, max_gap=continuation_gap
    )
    source_geojson = output_dir / "source_aligned_contours_pixels.geojson"
    contours.to_file(source_geojson, driver="GeoJSON")

    dissolved, dissolved_crossings = dissolve_and_resolve(
        preserved_rows, max_gap=continuation_gap
    )

    # The first grid was solved from the handful of traces that passed QC
    # against the point surface, because nothing better existed yet. Now whole
    # isolines do: assembled, harmonized to one level, and far longer. Rebuild
    # the surface on the long ones — length is what distinguishes an authored
    # contour from a leftover stroke, and a long contour also constrains far
    # more of the sheet.
    regrid = {"used": 0, "length_floor_px": 0.0, "applied": False, "backfilled": 0}
    dissolved_lengths = np.asarray(
        [LineString(chain["points"]).length for chain in dissolved]
    )
    if len(dissolved) >= 12:
        floor = float(dissolved_lengths.mean() * long_contour_fraction)
        regrid["length_floor_px"] = round(floor, 1)
        long_isolines = gpd.GeoDataFrame(
            [
                {"value_km": chain["value_km"], "geometry": LineString(chain["points"])}
                for chain, length in zip(dissolved, dissolved_lengths)
                if length >= floor
            ],
            geometry="geometry",
            crs="EPSG:3857",
        )
        if len(long_isolines) >= 10:
            grid, grid_quality = build_harmonic_grid(
                long_isolines.assign(constraint_weight=25.0),
                cell_size=max(15.0, min(target_size) / 250.0),
                blanking_distance=min(target_size) / 12.0,
                constraint_weight=25.0,
                support_contours=support,
            )
            reconstructed = extract_surface_contours(grid, interval=interval * 1000.0)
            grid_spacing = (
                float(np.median(np.diff(grid.x))) if len(grid.x) > 1 else 1.0
            )
            reconstructed, contour_cleanup = remove_unsupported_closed_contours(
                reconstructed,
                long_isolines,
                interval_m=interval * 1000.0,
                support_distance=grid_spacing * 4.0,
            )
            topology = contour_topology(reconstructed)
            regrid.update(used=len(long_isolines), applied=True)

            # Fragments the first, weak grid could not reach may sit cleanly on
            # this far better surface — the crimson unvalued pieces the eye
            # reads as miscoloured contour. Value them from it, then re-harmonize
            # and re-dissolve so both published layers carry the new levels.
            final_interpolator = RegularGridInterpolator(
                (grid.y, grid.x), grid.z, bounds_error=False, fill_value=np.nan
            )
            regrid["backfilled"] = backfill_from_grid(
                preserved_rows,
                final_interpolator,
                interval=interval,
                residual_limit=0.35,
                spread_limit=0.9,
                source="regrid_backfill",
            )
            if regrid["backfilled"]:
                unify_continuation_values(preserved_rows, max_gap=continuation_gap)
                dissolved, extra = dissolve_and_resolve(
                    preserved_rows, max_gap=continuation_gap
                )
                dissolved_crossings += extra

    preserved = gpd.GeoDataFrame(
        preserved_rows, geometry="geometry", crs="EPSG:3857"
    )
    preserved_geojson = output_dir / "digitized_source_contours_pixels.geojson"
    preserved.to_file(preserved_geojson, driver="GeoJSON")
    dissolved_frame = gpd.GeoDataFrame(
        [
            {
                "value_km": chain["value_km"],
                "value_m": chain["value_km"] * 1000.0,
                "member_trace_ids": json.dumps(chain["members"]),
                "fragments": len(chain["members"]),
                "geometry": LineString(chain["points"]),
            }
            for chain in dissolved
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    dissolved_geojson = output_dir / "digitized_isolines_by_level.geojson"
    dissolved_frame.to_file(dissolved_geojson, driver="GeoJSON")
    final_geojson = output_dir / "reconstructed_contours_pixels.geojson"
    reconstructed.to_file(final_geojson, driver="GeoJSON")
    grid_path = output_dir / "trace_guided_surface_pixels.npz"
    np.savez_compressed(grid_path, x=grid.x, y=grid.y, z=grid.z)

    with Image.open(source_path) as source:
        background = np.asarray(source.convert("L"))
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=120)
    preview_step = max(1, int(np.ceil(max(background.shape) / 3000.0)))
    axes[0].imshow(
        background[::preview_step, ::preview_step],
        cmap="gray",
        extent=(0, background.shape[1], background.shape[0], 0),
    )
    for item in assignments:
        points = np.asarray(item["geometry"].coords) / scale
        color = "#00a36c" if item["accepted"] else "#cc3344" if item["profile_suspect"] else "#999999"
        axes[0].plot(points[:, 0], points[:, 1], color=color, lw=1.0, alpha=0.9)
    axes[0].set_title(f"Source traces: {len(accepted)} accepted")
    axes[0].axis("off")
    masked = np.ma.masked_invalid(grid.z)
    fill = axes[1].pcolormesh(grid.x, grid.y, masked, shading="auto", cmap="turbo")
    reconstructed.plot(ax=axes[1], color="#111111", linewidth=0.65)
    contours.plot(ax=axes[1], color="white", linewidth=0.65, alpha=0.9)
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    axes[1].set_title("Trace-guided harmonic surface")
    axes[1].axis("off")
    fig.colorbar(fill, ax=axes[1], shrink=0.75, label="Depth, m")
    preview = output_dir / "trace_guided_surface_preview.png"
    fig.savefig(preview, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "traces": len(assignments),
        "accepted_traces": len(accepted),
        "accepted_rate": round(len(accepted) / max(1, len(assignments)), 4),
        "profile_suspects": sum(item["profile_suspect"] for item in assignments),
        "preserved_source_traces": len(preserved),
        "preserved_valued_traces": int(preserved["value_m"].notna().sum()),
        "grid_backfilled_traces": grid_backfilled,
        "neighbor_order_traces": neighbor_valued,
        "preserved_value_sources": {
            source: sum(row["value_source"] == source for row in preserved_rows)
            for source in sorted({row["value_source"] for row in preserved_rows})
        },
        "dissolved_isolines": len(dissolved),
        "dissolved_from_fragments": int(
            sum(len(chain["members"]) for chain in dissolved)
        ),
        "minimum_trace_length_px": round(minimum_trace_length, 2),
        "direct_labels": len(direct_labels),
        "value_source_distribution": {
            source: sum(
                1
                for item in accepted
                if (
                    "direct_ocr"
                    if item["direct_label"]
                    else "profile_order"
                    if item["profile_order"]
                    else "surface_interpolation"
                )
                == source
            )
            for source in ("direct_ocr", "profile_order", "surface_interpolation")
        },
        "profile_propagation": {
            "profile_lines": propagation["profile_lines"],
            "profiles_with_crossings": propagation["profiles_with_crossings"],
            "crossing_events": propagation["crossing_events"],
            "mark_anchors": propagation["mark_anchors"],
            "depth_mark_anchors_available": int(len(depth_marks))
            if depth_marks is not None
            else 0,
            "seed_traces": propagation["seed_traces"],
            "propagated_traces": len(propagation["propagated_values"]),
            "conflicting_traces": len(propagation["conflicts"]),
            "rounds": propagation["rounds"],
            "profile_points": len(propagation["points"]),
        },
        "crossing_rejections": crossing_rejections,
        "mark_outlier_rejections": mark_outlier_rejections,
        "crossing_trace_rejections": sorted(crossing_trace_ids),
        "profile_tick_rejections": len(tick_traces),
        "rehabilitated_profile_suspects": len(rehabilitated),
        "grid_constraint_contours": len(grid_contours),
        "grid_constraint_length_floor_px": round(constraint_length_floor, 1),
        "long_isoline_regrid": regrid,
        "dissolved_crossing_rejections": dissolved_crossings,
        "continuation_consensus": continuation,
        "depth_mark_grid_constraints": int(len(depth_marks))
        if depth_marks is not None
        else 0,
        "trace_image_size": list(trace_size),
        "surface_image_size": [round(target_size[0]), round(target_size[1])],
        "trace_to_surface_scale": [round(float(value), 6) for value in scale],
        "value_distribution": {
            str(value): sum(item["value_km"] == value for item in accepted)
            for value in sorted({item["value_km"] for item in accepted})
        },
        "grid_quality": grid_quality,
        "topology": topology,
        "contour_cleanup": contour_cleanup,
        "files": {
            "source_contours": str(source_geojson),
            "digitized_contours": str(preserved_geojson),
            "dissolved_isolines": str(dissolved_geojson),
            "final_contours": str(final_geojson),
            "grid": str(grid_path),
            "profile_points": str(profile_points_path),
            "preview": str(preview),
        },
    }
    (output_dir / "trace_guided_surface_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--point-surface", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--trace-image", type=Path)
    parser.add_argument("--labels", type=Path, nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(args.traces, args.point_surface, args.output_dir, interval=args.interval, trace_image_path=args.trace_image, label_paths=args.labels), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
