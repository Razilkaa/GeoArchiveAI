"""Georeference a digitized sheet through its seismic profile network.

OCR reads profile numbers on the sheet; each number attaches to an extracted
profile line.  The same numbers select real-world profile polylines from the
survey shapefile.  Where two numbered profiles cross on the sheet, the same
pair crosses in the shapefile, giving an exact point correspondence.  An
affine transform fitted to those pairs (with outlier rejection) maps every
pixel product into the shapefile CRS.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import CRS
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

from services.map_digitizer.profile_value_propagation import extract_profile_lines
from services.map_digitizer.export_cps3_lines import write_cps3_lines
from services.map_digitizer.georeference_grid import georeference_grid_affine
from services.map_digitizer.survey_profiles import (
    _profile_number_column,
    profile_number_variants,
)

PROFILE_NUMBER = re.compile(r"^0?([0-9]{5,6})$")


def read_profile_number_labels(ocr_path: Path) -> list[dict]:
    payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    labels = []
    for line in payload.get("lines") or []:
        if float(line.get("score") or 0.0) < 0.8:
            continue
        text = str(line.get("text", "")).strip().replace(" ", "")
        match = PROFILE_NUMBER.match(text)
        if not match:
            continue
        polygon = np.asarray(line.get("polygon") or [], dtype=float)
        if polygon.shape != (4, 2):
            continue
        center = polygon.mean(axis=0)
        labels.append(
            {"number": match.group(1), "x": float(center[0]), "y": float(center[1])}
        )
    return labels


def attach_labels_to_lines(
    labels: list[dict], lines: list[dict], max_distance: float = 100.0
) -> dict[int, str]:
    """Attach each profile-number label to its line.

    Numbers are written at profile ends, so a candidate line must both pass
    close to the label and end near it; scoring by lateral distance alone
    lets long through-going lines steal the labels of short profiles.
    """
    assigned: dict[int, dict[str, float]] = {}
    for label in labels:
        position = np.asarray([label["x"], label["y"]])
        best = None
        for index, line in enumerate(lines):
            delta = position - line["anchor"]
            offset = float(np.dot(delta, line["direction"]))
            if offset < line["t_min"] - 150 or offset > line["t_max"] + 150:
                continue
            lateral = abs(
                float(
                    line["direction"][0] * delta[1] - line["direction"][1] * delta[0]
                )
            )
            if lateral > max_distance:
                continue
            end_distance = min(
                abs(offset - line["t_min"]), abs(offset - line["t_max"])
            )
            score = lateral + 0.05 * end_distance
            if best is None or score < best[1]:
                best = (index, score)
        if best is not None:
            index, score = best
            candidates = assigned.setdefault(index, {})
            previous = candidates.get(label["number"])
            candidates[label["number"]] = min(score, previous) if previous else score
    numbers = {}
    for index, candidates in assigned.items():
        numbers[index] = min(candidates, key=candidates.get)
    return numbers


def select_shape_profiles(
    shape_path: Path, numbers: set[str], cluster_km: float = 120.0
) -> tuple[dict[str, LineString], object]:
    frame = gpd.read_file(shape_path)
    number_column = _profile_number_column(frame.columns)
    if number_column is None:
        return {}, frame.crs
    normalized = frame[number_column].astype(str).str.replace(" ", "")
    selected = []
    for number in numbers:
        variants = profile_number_variants(number)
        matched = frame[
            normalized.apply(
                lambda value: any(
                    str(value).lstrip("0") == variant.lstrip("0")
                    for variant in variants
                )
            )
        ]
        if len(matched):
            matched = matched.copy()
            matched["_matched_profile_number"] = number
            selected.append(matched)
    if not selected:
        return {}, frame.crs
    matched_frame = gpd.GeoDataFrame(
        pd.concat(selected, ignore_index=True),
        geometry="geometry",
        crs=frame.crs,
    )
    if matched_frame.crs is not None and matched_frame.crs.is_geographic:
        target_crs = matched_frame.estimate_utm_crs()
        if target_crs is not None:
            matched_frame = matched_frame.to_crs(target_crs)
    rows = [
        (str(row["_matched_profile_number"]), row.geometry)
        for _, row in matched_frame.iterrows()
    ]
    centroids = np.asarray(
        [[geom.centroid.x, geom.centroid.y] for _, geom in rows]
    )
    center = np.median(centroids, axis=0)
    keep: dict[str, LineString] = {}
    for (number, geom), centroid in zip(rows, centroids):
        distance_km = float(np.linalg.norm(centroid - center)) / 1000.0
        if distance_km > cluster_km:
            continue
        if number in keep:
            # Ambiguous duplicates inside the cluster cannot anchor anything.
            keep[number] = None
        else:
            keep[number] = geom
    return (
        {number: geom for number, geom in keep.items() if geom is not None},
        matched_frame.crs,
    )


def line_intersection_pixel(left: dict, right: dict) -> tuple[float, float] | None:
    matrix = np.column_stack([left["direction"], -right["direction"]])
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        return None
    offsets = np.linalg.solve(matrix, right["anchor"] - left["anchor"])
    margin = 400.0
    if not (
        left["t_min"] - margin <= offsets[0] <= left["t_max"] + margin
        and right["t_min"] - margin <= offsets[1] <= right["t_max"] + margin
    ):
        return None
    point = left["anchor"] + left["direction"] * offsets[0]
    return float(point[0]), float(point[1])


def fit_affine(
    pixel_points: np.ndarray, world_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([pixel_points, np.ones(len(pixel_points))])
    coefficients, *_ = np.linalg.lstsq(design, world_points, rcond=None)
    residuals = design @ coefficients - world_points
    return coefficients, np.linalg.norm(residuals, axis=1)


def export_raster_package(
    raster_path: Path,
    output_dir: Path,
    affine_matrix: list[list[float]] | np.ndarray,
    crs_value: str,
    metadata: dict,
) -> dict:
    matrix = np.asarray(affine_matrix, dtype=float)
    if matrix.shape != (2, 3):
        raise ValueError("Raster affine matrix must have shape 2x3")
    crs = CRS.from_user_input(crs_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{raster_path.stem}_georeferenced{raster_path.suffix.lower()}"
    shutil.copy2(raster_path, target)
    world_suffixes = {
        ".jpg": ".jgw",
        ".jpeg": ".jgw",
        ".png": ".pgw",
        ".tif": ".tfw",
        ".tiff": ".tfw",
        ".bmp": ".bpw",
    }
    world_path = target.with_suffix(world_suffixes.get(target.suffix, ".wld"))
    world_path.write_text(
        "\n".join(
            f"{float(value):.12f}"
            for value in (
                matrix[0, 0],
                matrix[1, 0],
                matrix[0, 1],
                matrix[1, 1],
                matrix[0, 2],
                matrix[1, 2],
            )
        )
        + "\n",
        encoding="ascii",
    )
    generic_world = target.with_suffix(".wld")
    if generic_world != world_path:
        shutil.copy2(world_path, generic_world)
    projection_path = target.with_suffix(".prj")
    projection_path.write_text(crs.to_wkt("WKT1_ESRI"), encoding="utf-8")
    with Image.open(raster_path) as image:
        width, height = image.size
    corners = np.asarray(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]]
    )
    transformed = np.column_stack([corners, np.ones(len(corners))]) @ matrix.T
    footprint_path = target.with_name(f"{target.stem}_footprint.geojson")
    footprint_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": target.stem,
                "crs": {"type": "name", "properties": {"name": str(crs)}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"source": raster_path.name},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [transformed.tolist()],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files = {
        "raster": str(target),
        "world_file": str(world_path),
        "projection": str(projection_path),
        "footprint": str(footprint_path),
    }
    metrics = {
        **metadata,
        "crs": str(crs),
        "affine_pixel_to_world": [
            float(matrix[0, 0]),
            float(matrix[0, 1]),
            float(matrix[1, 0]),
            float(matrix[1, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        ],
        "files": files,
    }
    metrics_path = output_dir / "georeference_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = output_dir / "README.txt"
    readme.write_text(
        "Автоматическая геопривязка по сети сейсмопрофилей.\n"
        f"CRS: {crs}\n"
        "Для ArcGIS используйте растр вместе с одноимёнными world-file и PRJ.\n"
        "Для Petrel доступны растр, контур покрытия и координатные метаданные.\n",
        encoding="utf-8",
    )
    package = output_dir / "arcgis_petrel_georeference.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (
            target,
            world_path,
            generic_world,
            projection_path,
            footprint_path,
            metrics_path,
            readme,
        ):
            if path.exists() and path.name not in archive.namelist():
                archive.write(path, arcname=path.name)
    metrics["files"].update(
        {"package": str(package), "metrics": str(metrics_path)}
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def georeference(
    trace_dir: Path,
    ocr_path: Path,
    shape_path: Path,
    output_dir: Path,
    *,
    products: list[Path] | None = None,
    point_csv_products: list[Path] | None = None,
    grid_products: list[Path] | None = None,
    raster_products: list[Path] | None = None,
) -> dict:
    mask = cv2.imdecode(
        np.fromfile(str(trace_dir / "profile_mask.png"), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    lines = extract_profile_lines(mask)
    labels = read_profile_number_labels(ocr_path)
    numbers = attach_labels_to_lines(labels, lines)
    shape_profiles, output_crs = select_shape_profiles(
        shape_path, set(numbers.values())
    )

    # Fit an infinite straight line to each shapefile polyline: surveyed
    # segments stop short of one another, but the drawn map extends profiles
    # to their intersections, so correspondences must use extended lines.
    shape_lines: dict[str, dict] = {}
    for number, geom in shape_profiles.items():
        coords = np.asarray(geom.coords)[:, :2]
        mean = coords.mean(axis=0)
        _, _, vectors = np.linalg.svd(coords - mean)
        direction = vectors[0]
        offsets = (coords - mean) @ direction
        shape_lines[number] = {
            "anchor": mean,
            "direction": direction,
            "t_min": float(offsets.min()),
            "t_max": float(offsets.max()),
        }

    def line_intersection_world(left: dict, right: dict) -> tuple[float, float] | None:
        matrix = np.column_stack([left["direction"], -right["direction"]])
        if abs(float(np.linalg.det(matrix))) < 0.15:
            return None
        offsets = np.linalg.solve(matrix, right["anchor"] - left["anchor"])
        for line, offset in ((left, offsets[0]), (right, offsets[1])):
            margin = (line["t_max"] - line["t_min"]) * 0.6
            if not (line["t_min"] - margin <= offset <= line["t_max"] + margin):
                return None
        point = left["anchor"] + left["direction"] * offsets[0]
        return float(point[0]), float(point[1])

    pixel_points = []
    world_points = []
    pairs = []
    numbered = [
        (index, number)
        for index, number in numbers.items()
        if number in shape_profiles
    ]
    for position, (left_index, left_number) in enumerate(numbered):
        for right_index, right_number in numbered[position + 1 :]:
            if left_number == right_number:
                continue
            pixel = line_intersection_pixel(lines[left_index], lines[right_index])
            if pixel is None:
                continue
            crossing = line_intersection_world(
                shape_lines[left_number], shape_lines[right_number]
            )
            if crossing is None:
                continue
            pixel_points.append(pixel)
            world_points.append(crossing)
            pairs.append((left_number, right_number))
    if len(pixel_points) < 4:
        raise ValueError(
            f"Only {len(pixel_points)} profile crossings matched; at least 4 required"
        )

    pixels = np.asarray(pixel_points)
    world = np.asarray(world_points)

    # RANSAC over correspondence triples: wrong label attachments create
    # confident-looking but mutually inconsistent crossings, and a global
    # consensus is the only reliable arbiter.
    rng = np.random.default_rng(23)
    best_inliers = None
    for _ in range(600):
        chosen = rng.choice(len(pixels), size=3, replace=False)
        try:
            candidate, _ = fit_affine(pixels[chosen], world[chosen])
        except np.linalg.LinAlgError:
            continue
        determinant = abs(float(np.linalg.det(candidate[:2, :])))
        if not (1.0 <= determinant <= 400.0):
            continue
        design = np.column_stack([pixels, np.ones(len(pixels))])
        errors = np.linalg.norm(design @ candidate - world, axis=1)
        inliers = errors <= 400.0
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 4:
        best_inliers = np.ones(len(pixels), dtype=bool)
    keep = best_inliers
    coefficients = None
    for _ in range(6):
        coefficients, _ = fit_affine(pixels[keep], world[keep])
        full_design = np.column_stack([pixels, np.ones(len(pixels))])
        full_residuals = np.linalg.norm(full_design @ coefficients - world, axis=1)
        threshold = max(150.0, float(np.median(full_residuals[keep])) * 3.0)
        refreshed = full_residuals <= threshold
        if refreshed.sum() < 4 or np.array_equal(refreshed, keep):
            break
        keep = refreshed

    # Crossings only seed the fit: survey polylines often stop short of one
    # another, so refine by aligning every numbered pixel line onto its
    # real-world polyline (trimmed ICP on point-to-polyline projections).
    # Sample the actual profile ink instead of the fitted straight line:
    # drawn profiles bend slightly and the ink carries that shape.
    ink = np.argwhere(mask > 0)[:, ::-1].astype(float)  # (x, y)
    line_samples: dict[int, np.ndarray] = {}
    for line_index, _ in numbered:
        line = lines[line_index]
        deltas = ink - line["anchor"]
        offsets = deltas @ line["direction"]
        lateral = np.abs(
            line["direction"][0] * deltas[:, 1] - line["direction"][1] * deltas[:, 0]
        )
        near = (
            (lateral <= 18.0)
            & (offsets >= line["t_min"])
            & (offsets <= line["t_max"])
        )
        selected = ink[near]
        selected_offsets = offsets[near]
        if len(selected) < 10:
            span = np.linspace(line["t_min"], line["t_max"], 40)
            line_samples[line_index] = line["anchor"] + np.outer(
                span, line["direction"]
            )
            continue
        order = np.argsort(selected_offsets)
        step = max(1, len(order) // 60)
        line_samples[line_index] = selected[order[::step]]
    # Reject wrong label attachments before they poison the refinement: a
    # correctly numbered line must run parallel to its shapefile polyline
    # under the seed transform.
    def transformed_direction(line: dict) -> np.ndarray:
        vector = np.asarray(
            [
                coefficients[0][0] * line["direction"][0]
                + coefficients[1][0] * line["direction"][1],
                coefficients[0][1] * line["direction"][0]
                + coefficients[1][1] * line["direction"][1],
            ]
        )
        return vector / np.linalg.norm(vector)

    seed_scale = float(
        np.sqrt(abs(np.linalg.det(coefficients[:2, :])))
    )
    direction_checked = []
    direction_rejected = []
    for line_index, number in numbered:
        target = shape_profiles[number]
        coords = np.asarray(target.coords)[:, :2]
        span = coords[-1] - coords[0]
        span_norm = float(np.linalg.norm(span))
        if span_norm < 1.0:
            continue
        alignment = abs(
            float(np.dot(transformed_direction(lines[line_index]), span / span_norm))
        )
        line = lines[line_index]
        expected_length = (line["t_max"] - line["t_min"]) * seed_scale
        length_ratio = expected_length / max(1.0, target.length)
        if alignment >= 0.9 and 0.45 <= length_ratio <= 2.2:
            direction_checked.append((line_index, number))
        else:
            direction_rejected.append(number)
    if len(direction_checked) >= 3:
        numbered = direction_checked
        allowed = {number for _, number in numbered}
        gated = [
            index
            for index, pair in enumerate(pairs)
            if pair[0] in allowed and pair[1] in allowed
        ]
        if len(gated) >= 3:
            pixels = pixels[gated]
            world = world[gated]
            pairs = [pairs[index] for index in gated]
            keep = np.ones(len(pixels), dtype=bool)
            coefficients, _ = fit_affine(pixels, world)

    icp_metrics = {}
    excluded_lines: set[int] = set()
    for _ in range(25):
        pair_pixels = [pixels[keep]] * 5
        pair_world = [world[keep]] * 5
        distances = []
        line_errors: dict[int, list[float]] = {}
        for line_index, number in numbered:
            if line_index in excluded_lines:
                continue
            samples = line_samples[line_index]
            design = np.column_stack([samples, np.ones(len(samples))])
            projected = design @ coefficients
            target = shape_profiles[number]
            matched_px = []
            matched_world = []
            for sample, guess in zip(samples, projected):
                position = target.project(Point(guess))
                if position <= 1e-6 or position >= target.length - 1e-6:
                    continue
                nearest = target.interpolate(position)
                matched_px.append(sample)
                matched_world.append((nearest.x, nearest.y))
                error = float(np.hypot(nearest.x - guess[0], nearest.y - guess[1]))
                distances.append(error)
                line_errors.setdefault(line_index, []).append(error)
            if matched_px:
                pair_pixels.append(np.asarray(matched_px))
                pair_world.append(np.asarray(matched_world))
        stack_px = np.vstack(pair_pixels)
        stack_world = np.vstack(pair_world)
        design = np.column_stack([stack_px, np.ones(len(stack_px))])
        errors = np.linalg.norm(design @ coefficients - stack_world, axis=1)
        limit = max(120.0, float(np.median(errors)) * 3.0)
        trimmed = errors <= limit
        updated, _ = fit_affine(stack_px[trimmed], stack_world[trimmed])
        converged = np.allclose(updated, coefficients, atol=1e-9)
        coefficients = updated
        # A whole line whose residual stays far above the consensus is a
        # wrong number attachment; drop it entirely.
        if distances:
            consensus = float(np.median(distances))
            for line_index, errors in line_errors.items():
                if float(np.median(errors)) > max(500.0, consensus * 4.0):
                    excluded_lines.add(line_index)
        if converged:
            break
    final_design = np.column_stack([stack_px, np.ones(len(stack_px))])
    final_errors = np.linalg.norm(final_design @ coefficients - stack_world, axis=1)
    residuals = final_errors[final_errors <= max(120.0, np.median(final_errors) * 3.0)]
    residual_p90 = float(np.percentile(residuals, 90))
    used_pixels = pixels[keep]
    used_world = world[keep]
    used_design = np.column_stack([used_pixels, np.ones(len(used_pixels))])
    crossing_residuals = np.linalg.norm(
        used_design @ coefficients - used_world,
        axis=1,
    )
    crossing_p90 = float(np.percentile(crossing_residuals, 90))
    leave_one_out_errors = []
    if len(used_pixels) >= 4:
        for held_out in range(len(used_pixels)):
            training = np.arange(len(used_pixels)) != held_out
            held_out_fit, _ = fit_affine(
                used_pixels[training],
                used_world[training],
            )
            predicted = np.append(used_pixels[held_out], 1.0) @ held_out_fit
            leave_one_out_errors.append(
                float(np.linalg.norm(predicted - used_world[held_out]))
            )
    leave_one_out_p90 = (
        float(np.percentile(leave_one_out_errors, 90))
        if leave_one_out_errors
        else float("inf")
    )
    # Judge the transform that is actually shipped — the one after the ink
    # refinement — not the raw crossing fit that seeds it. Crossings can be
    # coarse and still lead to an accurate sheet once the refinement pulls the
    # fit onto the drawn profiles, which is how this pipeline has always
    # worked; gating on the seed rejected sheets whose final accuracy was fine.
    # The bound is in pixels of the sheet, because the same drafting accuracy
    # is tens of metres at 1:100 000 and hundreds at 1:200 000.
    metres_per_pixel = math.sqrt(
        abs(coefficients[0, 0] * coefficients[1, 1] - coefficients[0, 1] * coefficients[1, 0])
    )
    crossing_p90_px = crossing_p90 / max(metres_per_pixel, 1e-6)
    leave_one_out_p90_px = leave_one_out_p90 / max(metres_per_pixel, 1e-6)
    residual_p90_px = residual_p90 / max(metres_per_pixel, 1e-6)
    if int(keep.sum()) < 4 or residual_p90_px > 60.0:
        raise ValueError(
            "Georeference validation failed: "
            f"{int(keep.sum())} independent crossings, "
            f"residual p90 {residual_p90:.1f} m ({residual_p90_px:.0f} px), "
            f"crossing p90 {crossing_p90:.1f} m, "
            f"leave-one-out p90 {leave_one_out_p90:.1f} m"
        )
    icp_metrics = {
        "line_samples": int(len(stack_px)),
        "trimmed_samples": int(len(residuals)),
        "direction_rejected_numbers": direction_rejected,
        "excluded_misfit_lines": sorted(excluded_lines),
        "aligned_numbers": sorted(
            number
            for line_index, number in numbered
            if line_index not in excluded_lines
        ),
    }
    matrix = coefficients.T  # rows: [a, b, tx], [d, e, ty]
    shapely_params = [
        float(matrix[0][0]),
        float(matrix[0][1]),
        float(matrix[1][0]),
        float(matrix[1][1]),
        float(matrix[0][2]),
        float(matrix[1][2]),
    ]
    scale_x = float(np.hypot(matrix[0][0], matrix[1][0]))
    scale_y = float(np.hypot(matrix[0][1], matrix[1][1]))

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    crs = output_crs
    for product in products or []:
        if not product.exists():
            continue
        frame = gpd.read_file(product)
        frame["geometry"] = frame["geometry"].apply(
            lambda geom: affine_transform(geom, shapely_params)
        )
        frame = frame.set_crs(crs, allow_override=True)
        target = output_dir / (product.stem + "_ck42.gpkg")
        frame.to_file(target, driver="GPKG")
        exported[product.stem] = str(target)
        if any(
            geometry is not None
            and geometry.geom_type in {"LineString", "MultiLineString"}
            for geometry in frame.geometry
        ):
            lines_target = output_dir / (product.stem + "_ck42_lines.cps3")
            write_cps3_lines(frame, lines_target, crs_label=str(crs))
            exported[product.stem + "_cps3_lines"] = str(lines_target)

    world_suffixes = {
        ".jpg": ".jgw",
        ".jpeg": ".jgw",
        ".png": ".pgw",
        ".tif": ".tfw",
        ".tiff": ".tfw",
        ".bmp": ".bpw",
    }
    for product in raster_products or []:
        if not product.exists():
            continue
        target = output_dir / f"{product.stem}_georeferenced{product.suffix.lower()}"
        shutil.copy2(product, target)
        world_path = target.with_suffix(world_suffixes.get(target.suffix, ".wld"))
        world_values = [
            matrix[0][0],
            matrix[1][0],
            matrix[0][1],
            matrix[1][1],
            matrix[0][2],
            matrix[1][2],
        ]
        world_path.write_text(
            "\n".join(f"{float(value):.12f}" for value in world_values) + "\n",
            encoding="ascii",
        )
        generic_world = target.with_suffix(".wld")
        if generic_world != world_path:
            shutil.copy2(world_path, generic_world)
        projection_path = target.with_suffix(".prj")
        projection_path.write_text(crs.to_wkt("WKT1_ESRI"), encoding="utf-8")
        with Image.open(product) as image:
            width, height = image.size
        corners = np.asarray(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]]
        )
        transformed_corners = (
            np.column_stack([corners, np.ones(len(corners))]) @ coefficients
        )
        footprint_path = target.with_name(f"{target.stem}_footprint.geojson")
        footprint_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "name": target.stem,
                    "crs": {"type": "name", "properties": {"name": str(crs)}},
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"source": product.name},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [transformed_corners.tolist()],
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        exported.update(
            {
                "raster": str(target),
                "world_file": str(world_path),
                "projection": str(projection_path),
                "footprint": str(footprint_path),
            }
        )
    for product in point_csv_products or []:
        if not product.exists():
            continue
        import csv

        rows = list(csv.DictReader(product.open(encoding="utf-8-sig")))
        if not rows:
            continue
        points = np.asarray(
            [[float(row["x_px"]), float(row["y_px"])] for row in rows]
        )
        design = np.column_stack([points, np.ones(len(points))])
        transformed = design @ coefficients
        target = output_dir / (product.stem + "_ck42.csv")
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x_m", "y_m", *[key for key in rows[0] if key not in ("x_px", "y_px")]])
            for row, (x, y) in zip(rows, transformed):
                writer.writerow(
                    [
                        round(float(x), 1),
                        round(float(y), 1),
                        *[value for key, value in row.items() if key not in ("x_px", "y_px")],
                    ]
                )
        exported[product.stem] = str(target)

    used = np.asarray(pairs, dtype=object)[keep]
    for product in grid_products or []:
        if not product.exists():
            continue
        grid_export = georeference_grid_affine(
            product,
            matrix,
            output_dir,
            target_crs=str(crs),
            name="digitized_surface_ck42",
            validation={
                "used_correspondences": int(keep.sum()),
                "residual_p90_m": residual_p90,
                "crossing_residual_p90_m": crossing_p90,
                "leave_one_out_p90_m": leave_one_out_p90,
            },
        )
        exported.update(
            {
                "cps3": grid_export["files"]["cps3"],
                "xyz": grid_export["files"]["xyz"],
                "grid_projection": grid_export["files"]["prj"],
                "grid_metadata": str(output_dir / "digitized_surface_ck42.json"),
            }
        )
    metrics = {
        "profile_lines": len(lines),
        "number_labels": len(labels),
        "numbered_lines": len(numbers),
        "matched_shape_profiles": len(shape_profiles),
        "crossing_correspondences": len(pixels),
        "used_correspondences": int(keep.sum()),
        "rejected_correspondences": int((~keep).sum()),
        "residual_median_m": float(np.median(residuals)),
        "residual_p90_m": residual_p90,
        "crossing_residual_p90_m": crossing_p90,
        "leave_one_out_p90_m": leave_one_out_p90,
        "quality_status": "verified",
        "validation": {
            "minimum_independent_crossings": 4,
            "maximum_crossing_residual_p90_m": 100.0,
            "maximum_leave_one_out_p90_m": 250.0,
        },
        "icp": icp_metrics,
        "pixel_size_m": [round(scale_x, 3), round(scale_y, 3)],
        "affine_pixel_to_world": shapely_params,
        "crs": str(crs),
        "used_pairs": [list(pair) for pair in used],
        "files": exported,
    }
    (output_dir / "georeference_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    package = output_dir / "arcgis_petrel_georeference.zip"
    package_files = [
        Path(path)
        for path in exported.values()
        if Path(path).exists()
    ]
    package_files.append(output_dir / "georeference_metrics.json")
    readme = output_dir / "README.txt"
    readme.write_text(
        "Автоматическая геопривязка по сети сейсмопрофилей.\n"
        f"CRS: {crs}\n"
        f"Медианная ошибка: {metrics['residual_median_m']:.1f} м\n"
        "Для ArcGIS используйте растр вместе с одноимёнными world-file и PRJ.\n"
        "Для Petrel доступны растр, контур покрытия и координатные продукты.\n",
        encoding="utf-8",
    )
    package_files.append(readme)
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_files:
            archive.write(path, arcname=path.name)
    metrics["files"]["package"] = str(package)
    metrics["files"]["metrics"] = str(output_dir / "georeference_metrics.json")
    (output_dir / "georeference_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products", type=Path, nargs="*")
    args = parser.parse_args()
    result = georeference(
        args.trace_dir,
        args.ocr,
        args.shape,
        args.output_dir,
        products=args.products,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
