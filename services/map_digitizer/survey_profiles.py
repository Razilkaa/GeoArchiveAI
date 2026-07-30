"""Use an archived survey network as a geometric prior for profile masking."""
from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def profile_number_variants(value: str) -> tuple[str, ...]:
    variants = [value]
    if len(value) == 6:
        # Volga–Ural drafting writes 088704 as 04-0887.
        variants.append(value[2:] + value[:2])
    return tuple(dict.fromkeys(variants))


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
        profile_id = next(
            (
                candidate
                for candidate in profile_number_variants(text)
                if candidate in valid_ids
            ),
            None,
        )
        if profile_id is None or float(item.get("score") or 0.0) < 0.8:
            continue
        polygon = np.asarray(item.get("polygon") or [], dtype=float)
        if polygon.shape != (4, 2):
            continue
        center = polygon.mean(axis=0) * scale
        edge = polygon[1] - polygon[0]
        labels.append(
            {
                "profile_id": profile_id,
                "center": center,
                "angle": math.degrees(math.atan2(edge[1], edge[0])) % 180.0,
                "score": float(item.get("score") or 0.0),
            }
        )
    return labels


def _attach_labels(
    labels: list[dict], candidates: list[dict], world_lines: dict[str, np.ndarray]
) -> list[dict]:
    """Pair profile-number labels with traced lines by voting on map rotation.

    A number printed along a profile often sits far from the fragment Hough
    actually recovered, so nearest-line matching either misses the pair or
    grabs a neighbouring profile. Every candidate pair implies one map
    rotation; the correct pairs all imply the same one, so the largest
    rotation-consistent set is the trustworthy anchor set.
    """
    pairs = []
    for label in labels:
        world_angle = _line_angle(world_lines[label["profile_id"]])
        for candidate in candidates:
            distance = _distance_to_infinite_line(
                label["center"], candidate["points"][0], candidate["points"][1]
            )
            angle_delta = _angle_distance(label["angle"], candidate["angle"])
            # Numbers written along their profile are the reliable anchors, so
            # the gate stays: loosening it to catch sheets that label a
            # vertical profile with horizontal text was measured to degrade the
            # fit where labels do run along the line. Those sheets are served
            # by the pattern match instead, which needs no numbers.
            if distance > 420.0 or angle_delta > 30.0:
                continue
            pairs.append(
                {
                    **label,
                    "candidate_id": candidate["id"],
                    "rotation": (candidate["angle"] + world_angle) % 180.0,
                    # Angle no longer gates the pair, but it still ranks it, so
                    # a number written along its profile keeps its precedence
                    # over one that merely happens to lie nearby.
                    "match_cost": distance + angle_delta * 1.5,
                }
            )
    if not pairs:
        return []

    best: list[dict] = []
    for pivot in pairs:
        group = [
            pair
            for pair in pairs
            if _angle_distance(pair["rotation"], pivot["rotation"]) <= 6.0
        ]
        distinct = {pair["profile_id"] for pair in group}
        best_distinct = {pair["profile_id"] for pair in best}
        if len(distinct) > len(best_distinct) or (
            len(distinct) == len(best_distinct)
            and sum(pair["match_cost"] for pair in group)
            < sum(pair["match_cost"] for pair in best)
        ):
            best = group

    chosen: dict[str, dict] = {}
    for pair in best:
        current = chosen.get(pair["profile_id"])
        if current is None or pair["match_cost"] < current["match_cost"]:
            chosen[pair["profile_id"]] = pair
    return list(chosen.values())


def projection_ink_coverage(
    world_lines: dict[str, np.ndarray],
    parameters: np.ndarray,
    ink: np.ndarray,
    *,
    minimum_inside: float = 0.5,
) -> tuple[float, int]:
    """Fraction of the projected network that actually lands on drawn ink.

    A transform can satisfy the anchors it was fitted from and still put the
    survey nowhere near the sheet, so agreement with those anchors proves
    nothing on its own. The sheet draws its own profiles, so a correct
    projection has to fall on that ink. Profiles outside the sheet are skipped:
    a survey covers more ground than any one map of it.
    """
    height, width = ink.shape
    scores = []
    for coordinates in world_lines.values():
        if len(coordinates) < 2:
            continue
        pixels = _inverse_transform(coordinates, parameters)
        span = float(np.linalg.norm(np.diff(pixels, axis=0), axis=1).sum())
        count = max(8, int(span / 25.0))
        index = np.linspace(0, len(pixels) - 1, count)
        samples = np.column_stack(
            [
                np.interp(index, np.arange(len(pixels)), pixels[:, 0]),
                np.interp(index, np.arange(len(pixels)), pixels[:, 1]),
            ]
        )
        inside = (
            (samples[:, 0] >= 0)
            & (samples[:, 0] < width)
            & (samples[:, 1] >= 0)
            & (samples[:, 1] < height)
        )
        if float(np.mean(inside)) < minimum_inside:
            continue
        visible = samples[inside]
        rows = np.clip(np.round(visible[:, 1]).astype(int), 0, height - 1)
        columns = np.clip(np.round(visible[:, 0]).astype(int), 0, width - 1)
        scores.append(float(np.mean(ink[rows, columns] > 0)))
    if not scores:
        return 0.0, 0
    return float(np.median(scores)), len(scores)


def _align_line_patterns(
    candidates: list[dict],
    world_lines: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    ink: np.ndarray,
) -> tuple[np.ndarray, float, int] | None:
    """Fit the sheet to the survey network by matching the two line patterns.

    Reading profile numbers is not always possible: a sheet may label only a
    few of its profiles, and those few can run parallel, which pins no
    rotation. The networks themselves are distinctive, though, so two guessed
    correspondences propose a transform and the rest of the network votes on
    it. Numbers then only have to confirm a fit, not produce one.
    """
    if len(candidates) < 3 or len(world_lines) < 3:
        return None

    world = []
    for coordinates in world_lines.values():
        center, normal = _fitted_line(coordinates)
        world.append(
            {
                "center": center,
                "normal": normal,
                "angle": _line_angle(coordinates),
                "length": float(np.linalg.norm(coordinates[-1] - coordinates[0])),
            }
        )
    # Anchor the search on the longest lines: they carry the most reliable
    # bearing, and a wrong pairing of two long lines fails the vote loudly.
    strong_pixels = sorted(candidates, key=lambda item: -item["length"])[:14]
    strong_world = sorted(world, key=lambda item: -item["length"])[:22]
    diagonal = float(np.hypot(*image_shape))

    # A sheet depicts this survey, so its footprint has to resemble the
    # survey's own. Votes alone do not catch a transform that is right in
    # bearing but wrong in scale, and such a fit would mask real contours.
    survey_points = np.vstack([coords for coords in world_lines.values()])
    survey_span = float(
        np.hypot(*(survey_points.max(axis=0) - survey_points.min(axis=0)))
    )
    corners = np.array(
        [
            [0.0, 0.0],
            [float(image_shape[1]), 0.0],
            [0.0, float(image_shape[0])],
            [float(image_shape[1]), float(image_shape[0])],
        ]
    )

    best: tuple[float, float, np.ndarray, int] | None = None
    for pixel_a, pixel_b in combinations(strong_pixels, 2):
        if _angle_distance(pixel_a["angle"], pixel_b["angle"]) < 20.0:
            continue
        for world_a in strong_world:
            for world_b in strong_world:
                if world_a is world_b:
                    continue
                if _angle_distance(world_a["angle"], world_b["angle"]) < 20.0:
                    continue
                rotations = [
                    (pixel_a["angle"] + world_a["angle"]) % 180.0,
                    (pixel_b["angle"] + world_b["angle"]) % 180.0,
                ]
                if _angle_distance(rotations[0], rotations[1]) > 8.0:
                    continue
                doubled = np.radians(np.asarray(rotations)) * 2.0
                angle = math.atan2(
                    float(np.sin(doubled).mean()), float(np.cos(doubled).mean())
                ) / 2.0
                cosine, sine = math.cos(angle), math.sin(angle)
                unit = np.array([[cosine, sine], [sine, -cosine]])
                rows, targets = [], []
                for pixel, world_line in ((pixel_a, world_a), (pixel_b, world_b)):
                    midpoint = pixel["points"].mean(axis=0)
                    normal = world_line["normal"]
                    rows.append(
                        [normal[0], normal[1], float(normal @ (unit @ midpoint))]
                    )
                    targets.append(float(normal @ world_line["center"]))
                try:
                    tx, ty, fitted = np.linalg.lstsq(
                        np.asarray(rows), np.asarray(targets), rcond=None
                    )[0]
                except np.linalg.LinAlgError:
                    continue
                if not 2.0 <= abs(fitted) <= 60.0:
                    continue
                parameters = np.array([tx, ty, math.log(abs(fitted)), angle])
                sheet = _transform(corners, parameters)
                sheet_span = float(np.hypot(*(sheet.max(axis=0) - sheet.min(axis=0))))
                if not 0.5 <= sheet_span / max(1.0, survey_span) <= 2.5:
                    continue
                # The vote tolerance has to live in world units: a fraction of
                # the sheet's own diagonal, not a fixed metre count, or it is
                # sub-pixel on a large scan and nothing can ever agree.
                tolerance = max(150.0, 0.004 * diagonal * abs(fitted))
                votes = 0
                errors = []
                for pixel in candidates:
                    projected = _transform(pixel["points"], parameters)
                    center = projected.mean(axis=0)
                    angle_pixel = _line_angle(projected)
                    nearest = None
                    for world_line in world:
                        if _angle_distance(angle_pixel, world_line["angle"]) > 10.0:
                            continue
                        offset = abs(
                            float(world_line["normal"] @ (center - world_line["center"]))
                        )
                        if nearest is None or offset < nearest:
                            nearest = offset
                    if nearest is not None and nearest <= tolerance:
                        votes += 1
                        errors.append(nearest)
                if votes < max(4, len(candidates) // 6):
                    continue
                # Votes only say the hypothesis is self-consistent. What decides
                # between hypotheses is whether the projected network lands on
                # the profiles the sheet actually draws.
                coverage, inside = projection_ink_coverage(
                    world_lines, parameters, ink
                )
                if inside < 3:
                    continue
                score = float(np.median(errors))
                if best is None or coverage > best[0]:
                    best = (coverage, score, parameters, votes)
    if best is None:
        return None
    coverage, score, parameters, votes = best
    return parameters, score, votes


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


def _profile_number_column(columns) -> str | None:
    """Locate the profile-number attribute whatever its spelling.

    The regional inventories do not agree on case: the Yakutia set writes
    ``N_PROF``, the Volga-Ural one ``N_prof``.
    """
    for column in columns:
        if str(column).strip().casefold() in {"n_prof", "n_profil", "nprof"}:
            return str(column)
    return None


def find_survey_shape(inventory_id: str, shapes_dir: Path) -> Path | None:
    """Pick the regional survey inventory that actually holds this report.

    Reports arrive from different regions and each region ships its own
    shapefile, so the right one is the one containing the inventory number
    rather than a fixed path.
    """
    import pyogrio

    for candidate in sorted(shapes_dir.glob("*.shp")):
        try:
            info = pyogrio.read_info(candidate)
        except Exception:
            continue
        fields = [str(field) for field in info["fields"]]
        if "N_RGF" not in fields or _profile_number_column(fields) is None:
            continue
        try:
            found = pyogrio.read_dataframe(
                candidate,
                columns=["N_RGF"],
                read_geometry=False,
                where=f"N_RGF = '{inventory_id}'",
            )
        except Exception:
            continue
        if len(found):
            return candidate
    return None


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
    if inventory.crs is not None and inventory.crs.is_geographic and len(inventory):
        target_crs = inventory.estimate_utm_crs()
        if target_crs is not None:
            inventory = inventory.to_crs(target_crs)
    number_column = _profile_number_column(inventory.columns)
    if number_column is None:
        return {
            "status": "not_available",
            "reason": "profile_number_attribute_missing",
            "mask": np.zeros_like(linework),
        }
    world_lines = {}
    for _, row in inventory.iterrows():
        profile_id = str(row[number_column]).strip()
        if row.geometry is None or profile_id in {"", "nan", "None"}:
            continue
        geometry = row.geometry
        if geometry.geom_type == "MultiLineString":
            geometry = max(geometry.geoms, key=lambda part: part.length)
        if geometry.geom_type != "LineString":
            continue
        # One profile can arrive as several survey segments; the longest one
        # anchors the fit best and the rest of the line is projected anyway.
        coordinates = np.asarray(geometry.coords, dtype=float)[:, :2]
        existing = world_lines.get(profile_id)
        if existing is None or len(coordinates) > len(existing):
            world_lines[profile_id] = coordinates
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
    matches = _attach_labels(labels, candidates, world_lines)
    distinct_profiles = {item["profile_id"] for item in matches}
    distinct_candidates = {item["candidate_id"] for item in matches}
    candidate_lookup = {item["id"]: item for item in candidates}
    angles = [candidate_lookup[item]["angle"] for item in distinct_candidates]
    spread = max((_angle_distance(left, right) for left in angles for right in angles), default=0.0)
    # The spread guards against fitting a rotation from parallel lines alone.
    # It stays a real guard at 15 degrees, and the alignment RMS check below is
    # what actually rejects a badly conditioned fit.
    anchored = (
        len(distinct_profiles) >= 3
        and len(distinct_candidates) >= 3
        and spread >= 15.0
    )
    # The projected network only ever masks ink that already lies inside its
    # corridor, and a polyline is dropped only when most of it stays there, so
    # a few pixels of alignment slack cost nothing while rejecting the fit
    # outright loses the whole prior.
    maximum_rms_px = 30.0
    pattern_votes = 0
    parameters = None
    rms_m = float("inf")
    if anchored:
        parameters, rms_m = _fit_transform(matches, candidates, world_lines)
        if rms_m / math.exp(float(parameters[2])) > maximum_rms_px:
            parameters = None
    if parameters is None:
        # Either too few numbers were read, or the ones read placed the sheet
        # badly. The two networks are distinctive enough to match as patterns,
        # which needs no numbers at all.
        pattern = _align_line_patterns(
            candidates, world_lines, linework.shape, linework
        )
        if pattern is None:
            return {
                "status": "review",
                "reason": "insufficient_profile_anchors",
                "labels": len(labels),
                "matches": len(matches),
                "angle_spread_deg": round(spread, 2),
                "mask": np.zeros_like(linework),
            }
        parameters, rms_m, pattern_votes = pattern
        anchored = False
    scale_m_per_px = math.exp(float(parameters[2]))
    rms_px = rms_m / scale_m_per_px
    if rms_px > maximum_rms_px:
        return {
            "status": "review",
            "reason": "profile_alignment_error",
            "matches": len(matches),
            "pattern_votes": pattern_votes,
            "rms_m": round(rms_m, 2),
            "rms_px": round(rms_px, 2),
            "mask": np.zeros_like(linework),
        }

    # Final say goes to the sheet itself. A transform that satisfies its own
    # anchors can still land the survey off the drawing entirely, and masking
    # from it would erase contours while removing no profile at all.
    # Judge the projection at the accuracy the fit itself claims. A sheet is
    # drawn and scanned with distortion the transform cannot absorb, so a
    # correct alignment still sits tens of pixels off the ink; asking it to
    # land within a stroke width condemns every fit, right or wrong.
    tolerance = int(max(25.0, min(rms_px * 2.0, 80.0)))
    ink_coverage, profiles_inside = projection_ink_coverage(
        world_lines,
        parameters,
        cv2.dilate(
            linework,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1)
            ),
        ),
    )
    if profiles_inside < 3 or ink_coverage < 0.5:
        return {
            "status": "review",
            "reason": "projection_misses_drawn_profiles",
            "matches": len(matches),
            "pattern_votes": pattern_votes,
            "rms_px": round(rms_px, 2),
            "projected_profiles_inside_sheet": profiles_inside,
            "projection_ink_coverage": round(ink_coverage, 3),
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
    # The corridor has to be as wide as the alignment is uncertain. A six-pixel
    # corridor around a projection that sits tens of pixels off the drawing
    # confirms almost no ink, which is why an accepted mask still removed
    # nothing. Widening it costs nothing in contours: only ink inside the
    # corridor is masked at all, and a polyline is dropped only when most of
    # its length lies there, which a contour merely crossing never does.
    corridor_radius = int(np.clip(round(rms_px * 1.5), 8, 60))
    corridor = cv2.dilate(
        projected,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (corridor_radius * 2 + 1, corridor_radius * 2 + 1)
        ),
    )
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
        "fit_source": "profile_numbers" if anchored else "line_pattern",
        "pattern_votes": pattern_votes,
        "projection_ink_coverage": round(ink_coverage, 3),
        "projected_profiles_inside_sheet": profiles_inside,
        "corridor_radius_px": corridor_radius,
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
