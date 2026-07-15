"""Provisional georeferencing of sheet 23 from its seismic profile network."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import LineString

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def fitted_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0:
        direction *= -1
    normal = np.array([-direction[1], direction[0]])
    return center, direction, normal


def line_angle(points: np.ndarray) -> float:
    direction = fitted_line(points)[1]
    return math.degrees(math.atan2(direction[1], direction[0])) % 180.0


def transform_points(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    tx, ty, log_scale, angle = parameters
    scale = math.exp(log_scale)
    cosine, sine = math.cos(angle), math.sin(angle)
    # Map pixels have Y pointing down, hence the orientation-reversing matrix.
    matrix = scale * np.array([[cosine, sine], [sine, -cosine]])
    return points @ matrix.T + np.array([tx, ty])


def solve_similarity(
    pixel_lines: dict[int, np.ndarray],
    world_lines: dict[str, np.ndarray],
    pairs: list[tuple[int, str]],
) -> tuple[np.ndarray, float]:
    def residuals(parameters: np.ndarray) -> np.ndarray:
        result = []
        for pixel_id, profile_id in pairs:
            samples = np.linspace(pixel_lines[pixel_id][0], pixel_lines[pixel_id][-1], 7)
            center, _, normal = fitted_line(world_lines[profile_id])
            result.extend((transform_points(samples, parameters) - center) @ normal)
        return np.asarray(result)

    initial = np.array([800_000.0, 7_000_000.0, math.log(12.5), math.radians(7.0)])
    solution = least_squares(
        residuals,
        initial,
        loss="soft_l1",
        f_scale=250.0,
        max_nfev=4_000,
        x_scale="jac",
    )
    rms = float(np.sqrt(np.mean(residuals(solution.x) ** 2)))
    return solution.x, rms


def candidate_solutions(
    pixel_lines: dict[int, np.ndarray],
    pixel_meta: dict[int, dict],
    world_lines: dict[str, np.ndarray],
    exact_pairs: list[tuple[int, str]],
) -> list[dict]:
    first_id, second_id = (item[0] for item in exact_pairs[:2])
    first = LineString(pixel_lines[first_id])
    second = LineString(pixel_lines[second_id])
    anchor_pixel_angle = np.mean([pixel_meta[first_id]["angle_deg"], pixel_meta[second_id]["angle_deg"]])
    anchor_world_angle = np.mean([line_angle(world_lines[item[1]]) for item in exact_pairs[:2]])

    pixel_crossings = [
        line_id
        for line_id, points in pixel_lines.items()
        if line_id not in {first_id, second_id}
        and angle_distance(pixel_meta[line_id]["angle_deg"], anchor_pixel_angle) > 35.0
        and LineString(points).buffer(25).intersects(first)
        and LineString(points).buffer(25).intersects(second)
    ]
    world_crossings = [
        profile_id
        for profile_id, points in world_lines.items()
        if profile_id not in {item[1] for item in exact_pairs}
        and angle_distance(line_angle(points), anchor_world_angle) > 35.0
    ]

    ranked = []
    for pixel_id in pixel_crossings:
        for profile_id in world_crossings:
            pairs = exact_pairs[:2] + [(pixel_id, profile_id)]
            parameters, rms = solve_similarity(pixel_lines, world_lines, pairs)
            scale = math.exp(parameters[2])
            if not 5.0 <= scale <= 25.0:
                continue
            ranked.append(
                {
                    "pixel_line_id": pixel_id,
                    "profile_id": profile_id,
                    "rms_m": round(rms, 2),
                    "scale_m_per_px": round(scale, 4),
                    "rotation_deg": round(math.degrees(parameters[3]), 4),
                    "parameters": parameters.tolist(),
                }
            )
    return sorted(ranked, key=lambda item: item["rms_m"])


def transformed_line(points: np.ndarray, parameters: np.ndarray) -> LineString:
    return LineString(transform_points(points, parameters))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--isolines", type=Path, required=True)
    parser.add_argument("--gpkg", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()

    lines_payload = json.loads(args.lines.read_text(encoding="utf-8"))
    pixel_meta = {line["id"]: line for line in lines_payload["profile_candidates"]}
    pixel_lines = {
        line_id: np.asarray([line["p0"], line["p1"]], dtype=float)
        for line_id, line in pixel_meta.items()
    }
    matches = json.loads(args.matches.read_text(encoding="utf-8"))
    exact_pairs = [
        (label["line_id"], label["inventory_match"])
        for label in matches["labels"]
        if label["status"] == "exact_anchor"
    ]
    if len(exact_pairs) < 2:
        raise SystemExit("At least two exact labelled profiles are required")

    inventory = gpd.read_file(args.inventory)
    inventory["N_PROF"] = inventory["N_PROF"].astype(str).str.strip()
    world_lines = {
        row.N_PROF: np.asarray(row.geometry.coords, dtype=float)[:, :2]
        for _, row in inventory.iterrows()
        if row.N_PROF and row.N_PROF != "nan"
    }
    ranked = candidate_solutions(pixel_lines, pixel_meta, world_lines, exact_pairs)
    if not ranked:
        raise SystemExit("No geometrically plausible cross-profile anchor was found")
    selected = ranked[0]
    parameters = np.asarray(selected.pop("parameters"))
    inferred_pair = (selected["pixel_line_id"], selected["profile_id"])
    anchor_pairs = exact_pairs[:2] + [inferred_pair]

    profile_rows = []
    for line_id, points in pixel_lines.items():
        profile_rows.append(
            {
                "pixel_id": line_id,
                "support": pixel_meta[line_id]["support"],
                "length_px": pixel_meta[line_id]["length_px"],
                "angle_px": pixel_meta[line_id]["angle_deg"],
                "anchor": next((name for candidate, name in anchor_pairs if candidate == line_id), None),
                "geometry": transformed_line(points, parameters),
            }
        )
    profiles = gpd.GeoDataFrame(profile_rows, geometry="geometry", crs=inventory.crs)

    iso_payload = json.loads(args.isolines.read_text(encoding="utf-8"))
    iso_rows = []
    for feature in iso_payload["features"]:
        # Existing pixel GeoJSON uses GIS-style negative Y; restore image Y first.
        pixel_points = np.asarray([[x, -y] for x, y in feature["geometry"]["coordinates"]])
        iso_rows.append(
            {
                **feature["properties"],
                "geometry": transformed_line(pixel_points, parameters),
            }
        )
    isolines = gpd.GeoDataFrame(iso_rows, geometry="geometry", crs=inventory.crs)

    anchors = inventory[inventory["N_PROF"].isin([name for _, name in anchor_pairs])].copy()
    anchors["anchor_type"] = anchors["N_PROF"].map(
        {name: ("exact_label" if (line_id, name) in exact_pairs else "geometry_inferred") for line_id, name in anchor_pairs}
    )
    args.gpkg.unlink(missing_ok=True)
    profiles.to_file(args.gpkg, layer="profile_candidates", driver="GPKG")
    isolines.to_file(args.gpkg, layer="isolines", driver="GPKG")
    anchors.to_file(args.gpkg, layer="inventory_anchors", driver="GPKG")

    second_rms = ranked[1]["rms_m"] if len(ranked) > 1 else None
    qc = {
        "status": "review",
        "crs": str(inventory.crs),
        "method": "orientation-reversing similarity from two exact labels and one network match",
        "exact_anchors": [{"pixel_line_id": line_id, "profile_id": name} for line_id, name in exact_pairs[:2]],
        "inferred_anchor": {"pixel_line_id": inferred_pair[0], "profile_id": inferred_pair[1]},
        "selected_solution": selected,
        "second_best_rms_m": second_rms,
        "ambiguity_ratio": round(second_rms / selected["rms_m"], 3) if second_rms else None,
        "profile_candidates_exported": len(profiles),
        "isolines_exported": len(isolines),
        "warning": "The third anchor is inferred from network geometry; verify against a labelled cross-profile before production use.",
        "ranked_solutions": [{k: v for k, v in item.items() if k != "parameters"} for item in ranked[:10]],
    }
    args.qc.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(13, 11), dpi=150)
    inventory.plot(ax=ax, color="#b8b8b8", linewidth=1.2, label="inventory")
    isolines.plot(ax=ax, color="#2878b5", linewidth=0.6, alpha=0.5, label="digitized isolines")
    profiles.plot(ax=ax, color="#d73027", linewidth=0.8, alpha=0.65, label="detected straight lines")
    anchors.plot(ax=ax, color="#111111", linewidth=3.0, label="anchors")
    for _, row in anchors.iterrows():
        point = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(row.N_PROF, (point.x, point.y), fontsize=8, weight="bold")
    ax.set_aspect("equal")
    min_x, min_y, max_x, max_y = isolines.total_bounds
    margin_x = max((max_x - min_x) * 0.08, 1_000.0)
    margin_y = max((max_y - min_y) * 0.08, 1_000.0)
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    ax.set_title(
        f"Sheet 23 provisional georeference | EPSG:2509 | RMS {selected['rms_m']:.0f} m | REVIEW"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.preview, facecolor="white", bbox_inches="tight")
    print(
        f"selected pixel {inferred_pair[0]} -> {inferred_pair[1]}, RMS={selected['rms_m']} m; "
        f"wrote {args.gpkg}"
    )


if __name__ == "__main__":
    main()
