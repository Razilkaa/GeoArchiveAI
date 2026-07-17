"""Use an archived survey network as a geometric prior for profile masking."""
from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def _angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _line_angle(points: np.ndarray) -> float:
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    direction = vectors[0]
    return math.degrees(math.atan2(direction[1], direction[0])) % 180.0


def _fitted_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    direction = vectors[0]
    normal = np.array([-direction[1], direction[0]])
    return center, normal


def _distance_to_infinite_line(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    offset = point - start
    determinant = float(delta[0] * offset[1] - delta[1] * offset[0])
    return abs(determinant) / max(1.0, float(np.linalg.norm(delta)))


def _consolidate_hough(lines: np.ndarray | None, minimum_length: float) -> list[dict]:
    if lines is None:
        return []
    raw = []
    for values in lines.reshape(-1, 4):
        start = values[:2].astype(float)
        end = values[2:].astype(float)
        length = float(np.linalg.norm(end - start))
        if length < minimum_length:
            continue
        angle = _line_angle(np.vstack([start, end]))
        direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        normal = np.array([-direction[1], direction[0]])
        raw.append(
            {
                "start": start,
                "end": end,
                "angle": angle,
                "rho": float(normal @ ((start + end) / 2.0)),
                "length": length,
            }
        )

    parent = list(range(len(raw)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(raw):
        for right_index in range(left_index + 1, len(raw)):
            right = raw[right_index]
            if _angle_distance(left["angle"], right["angle"]) > 3.0:
                continue
            if abs(left["rho"] - right["rho"]) > 24.0:
                continue
            left_direction = (left["end"] - left["start"]) / left["length"]
            left_interval = sorted([left["start"] @ left_direction, left["end"] @ left_direction])
            right_interval = sorted([right["start"] @ left_direction, right["end"] @ left_direction])
            if max(left_interval[0], right_interval[0]) > min(left_interval[1], right_interval[1]) + 180.0:
                continue
            left_root, right_root = find(left_index), find(right_index)
            if left_root != right_root:
                parent[right_root] = left_root

    groups: dict[int, list[dict]] = {}
    for index, segment in enumerate(raw):
        groups.setdefault(find(index), []).append(segment)
    result = []
    for group in groups.values():
        points = np.vstack([[item["start"], item["end"]] for item in group])
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
        direction = vectors[0]
        projections = (points - center) @ direction
        start = center + projections.min() * direction
        end = center + projections.max() * direction
        length = float(np.linalg.norm(end - start))
        if length >= minimum_length:
            result.append(
                {
                    "id": len(result),
                    "points": np.vstack([start, end]),
                    "angle": _line_angle(np.vstack([start, end])),
                    "length": length,
                    "support": len(group),
                }
            )
    return result


def _profile_labels(ocr_payload: dict, valid_ids: set[str], scale: float) -> list[dict]:
    labels = []
    for item in ocr_payload.get("lines") or []:
        text = re.sub(r"\D", "", str(item.get("text") or ""))
        if text not in valid_ids or float(item.get("score") or 0.0) < 0.8:
            continue
        polygon = np.asarray(item.get("polygon") or [], dtype=float)
        if polygon.shape != (4, 2):
            continue
        center = polygon.mean(axis=0) * scale
        edge = polygon[1] - polygon[0]
        labels.append(
            {
                "profile_id": text,
                "center": center,
                "angle": math.degrees(math.atan2(edge[1], edge[0])) % 180.0,
                "score": float(item.get("score") or 0.0),
            }
        )
    return labels


def _attach_labels(labels: list[dict], candidates: list[dict]) -> list[dict]:
    matches = []
    for label in labels:
        ranked = []
        for candidate in candidates:
            distance = _distance_to_infinite_line(
                label["center"], candidate["points"][0], candidate["points"][1]
            )
            angle_delta = _angle_distance(label["angle"], candidate["angle"])
            if distance <= 45.0 and angle_delta <= 20.0:
                ranked.append((distance + angle_delta * 1.5, candidate))
        if ranked:
            cost, candidate = min(ranked, key=lambda item: item[0])
            matches.append({**label, "candidate_id": candidate["id"], "match_cost": cost})
    deduplicated = {}
    for match in matches:
        key = (match["profile_id"], match["candidate_id"])
        if key not in deduplicated or match["score"] > deduplicated[key]["score"]:
            deduplicated[key] = match
    return list(deduplicated.values())


def _transform(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    tx, ty, log_scale, angle = parameters
    scale = math.exp(float(log_scale))
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    matrix = scale * np.array([[cosine, sine], [sine, -cosine]])
    return points @ matrix.T + np.array([tx, ty])


def _fit_transform(matches: list[dict], candidates: list[dict], world_lines: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    candidate_by_id = {item["id"]: item for item in candidates}
    def residuals(parameters: np.ndarray, selected: list[dict]) -> np.ndarray:
        values = []
        for match in selected:
            pixel_line = candidate_by_id[match["candidate_id"]]["points"]
            samples = np.linspace(pixel_line[0], pixel_line[-1], 7)
            world_line = world_lines[match["profile_id"]]
            center, normal = _fitted_line(world_line)
            values.extend((_transform(samples, parameters) - center) @ normal)
        return np.asarray(values)

    def solve(selected: list[dict]) -> np.ndarray:
        rotation_hints = [
            math.radians(
                candidate_by_id[item["candidate_id"]]["angle"]
                + _line_angle(world_lines[item["profile_id"]])
            )
            for item in selected
        ]
        doubled = np.asarray(rotation_hints) * 2.0
        angle = math.atan2(np.sin(doubled).mean(), np.cos(doubled).mean()) / 2.0
        cosine, sine = math.cos(angle), math.sin(angle)
        unit_matrix = np.array([[cosine, sine], [sine, -cosine]])
        rows, targets = [], []
        for match in selected:
            pixel_line = candidate_by_id[match["candidate_id"]]["points"]
            world_center, world_normal = _fitted_line(world_lines[match["profile_id"]])
            for point in np.linspace(pixel_line[0], pixel_line[-1], 7):
                rows.append(
                    [world_normal[0], world_normal[1], float(world_normal @ (unit_matrix @ point))]
                )
                targets.append(float(world_normal @ world_center))
        tx, ty, fitted_scale = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)[0]
        if fitted_scale < 0:
            fitted_scale = abs(fitted_scale)
            angle = (angle + math.pi) % (2.0 * math.pi)
        fitted_scale = float(np.clip(fitted_scale, 2.0, 40.0))
        return np.array([tx, ty, math.log(fitted_scale), angle])

    subsets = [list(items) for items in combinations(matches, 3)] if len(matches) > 3 else [matches]
    ranked = []
    for subset in subsets:
        pixel_angles = [candidate_by_id[item["candidate_id"]]["angle"] for item in subset]
        if max((_angle_distance(a, b) for a in pixel_angles for b in pixel_angles), default=0.0) < 25.0:
            continue
        parameters = solve(subset)
        per_match = []
        for match in matches:
            error = residuals(parameters, [match])
            per_match.append(float(np.sqrt(np.mean(error**2))))
        threshold = max(300.0, math.exp(float(parameters[2])) * 25.0)
        inlier_indices = [index for index, error in enumerate(per_match) if error <= threshold]
        ranked.append(
            (
                -len(inlier_indices),
                float(np.median([per_match[index] for index in inlier_indices])) if inlier_indices else np.inf,
                parameters,
                inlier_indices,
            )
        )
    if not ranked:
        parameters = solve(matches)
        values = residuals(parameters, matches)
        return parameters, float(np.sqrt(np.mean(values**2)))
    _, _, parameters, inlier_indices = min(ranked, key=lambda item: (item[0], item[1]))
    if len(inlier_indices) >= 3:
        parameters = solve([matches[index] for index in inlier_indices])
        values = residuals(parameters, [matches[index] for index in inlier_indices])
    else:
        values = residuals(parameters, matches)
    return parameters, float(np.sqrt(np.mean(values**2)))


def _inverse_transform(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    tx, ty, log_scale, angle = parameters
    scale = math.exp(float(log_scale))
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    matrix = scale * np.array([[cosine, sine], [sine, -cosine]])
    return (points - np.array([tx, ty])) @ np.linalg.inv(matrix).T


def build_survey_profile_mask(
    linework: np.ndarray,
    ocr_payload: dict,
    *,
    inventory_id: str,
    shape_path: Path,
    scale: float,
    output_dir: Path,
) -> dict:
    """Project inventory profiles to the scan and return a raster-confirmed mask."""
    import pyogrio

    inventory = pyogrio.read_dataframe(
        shape_path,
        where=f"N_RGF = '{inventory_id}'",
    )
    world_lines = {
        str(row.N_PROF).strip(): np.asarray(row.geometry.coords, dtype=float)[:, :2]
        for _, row in inventory.iterrows()
        if row.geometry is not None and str(row.N_PROF).strip() not in {"", "nan"}
    }
    if len(world_lines) < 3:
        return {"status": "not_available", "reason": "survey_inventory_not_found", "mask": np.zeros_like(linework)}

    raw_lines = cv2.HoughLinesP(
        linework,
        rho=1,
        theta=np.pi / 720,
        threshold=100,
        minLineLength=max(150, int(min(linework.shape) * 0.09)),
        maxLineGap=45,
    )
    candidates = _consolidate_hough(raw_lines, max(180, min(linework.shape) * 0.11))
    labels = _profile_labels(ocr_payload, set(world_lines), scale)
    matches = _attach_labels(labels, candidates)
    distinct_profiles = {item["profile_id"] for item in matches}
    distinct_candidates = {item["candidate_id"] for item in matches}
    candidate_lookup = {item["id"]: item for item in candidates}
    angles = [candidate_lookup[item]["angle"] for item in distinct_candidates]
    spread = max((_angle_distance(left, right) for left in angles for right in angles), default=0.0)
    if len(distinct_profiles) < 3 or len(distinct_candidates) < 3 or spread < 25.0:
        return {
            "status": "review",
            "reason": "insufficient_profile_anchors",
            "labels": len(labels),
            "matches": len(matches),
            "angle_spread_deg": round(spread, 2),
            "mask": np.zeros_like(linework),
        }

    parameters, rms_m = _fit_transform(matches, candidates, world_lines)
    scale_m_per_px = math.exp(float(parameters[2]))
    rms_px = rms_m / scale_m_per_px
    if rms_px > 22.0:
        return {
            "status": "review",
            "reason": "profile_alignment_error",
            "matches": len(matches),
            "rms_m": round(rms_m, 2),
            "rms_px": round(rms_px, 2),
            "mask": np.zeros_like(linework),
        }

    projected = np.zeros_like(linework)
    features = []
    for profile_id, world_line in world_lines.items():
        pixel_line = _inverse_transform(world_line, parameters)
        integer_points = np.round(pixel_line).astype(np.int32)
        cv2.polylines(projected, [integer_points], False, 255, 3, cv2.LINE_AA)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": pixel_line.round(2).tolist()},
                "properties": {"profile_id": profile_id, "source": "survey_inventory"},
            }
        )
    corridor = cv2.dilate(projected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    confirmed = cv2.bitwise_and(linework, corridor)
    confirmed = cv2.dilate(confirmed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", confirmed)[1].tofile(str(output_dir / "survey_profile_mask.png"))
    (output_dir / "survey_profiles_pixels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = {
        "status": "applied",
        "inventory_id": inventory_id,
        "inventory_profiles": len(world_lines),
        "ocr_profile_labels": len(labels),
        "matched_anchors": len(matches),
        "angle_spread_deg": round(spread, 2),
        "rms_m": round(rms_m, 2),
        "rms_px": round(rms_px, 2),
        "scale_m_per_px": round(scale_m_per_px, 4),
        "crs": str(inventory.crs),
        "transform": parameters.tolist(),
        "mask": str(output_dir / "survey_profile_mask.png"),
        "profiles": str(output_dir / "survey_profiles_pixels.geojson"),
    }
    (output_dir / "survey_profile_alignment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**metadata, "mask": confirmed}
