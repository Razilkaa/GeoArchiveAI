from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _cross(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_segment_intersection(
    a: list[float], b: list[float], c: list[float], d: list[float]
) -> bool:
    if (
        max(a[0], b[0]) < min(c[0], d[0])
        or max(c[0], d[0]) < min(a[0], b[0])
        or max(a[1], b[1]) < min(c[1], d[1])
        or max(c[1], d[1]) < min(a[1], b[1])
    ):
        return False
    first, second = _cross(a, b, c), _cross(a, b, d)
    third, fourth = _cross(c, d, a), _cross(c, d, b)
    return ((first > 0 > second) or (second > 0 > first)) and (
        (third > 0 > fourth) or (fourth > 0 > third)
    )


def crossing_isoline_pairs(features: list[dict[str, Any]]) -> list[list[int]]:
    isolines = [item for item in features if item.get("properties", {}).get("kind") == "isoline"]
    crossings = []
    for index, first in enumerate(isolines):
        first_points = first.get("geometry", {}).get("coordinates", [])
        for second in isolines[index + 1 :]:
            second_points = second.get("geometry", {}).get("coordinates", [])
            intersects = any(
                _proper_segment_intersection(a, b, c, d)
                for a, b in zip(first_points, first_points[1:])
                for c, d in zip(second_points, second_points[1:])
            )
            if intersects:
                crossings.append(
                    [int(first["properties"]["id"]), int(second["properties"]["id"])]
                )
    return crossings


def build_map_result(
    report_id: str,
    trace_dir: Path,
    output_path: Path,
    source_image: Path | None = None,
) -> dict[str, Any]:
    valued_path = trace_dir / "valued_isolines_23.json"
    crosscheck_path = trace_dir / "crosscheck_23.json"
    geojson_path = trace_dir / "isolines_23.geojson"
    preview_path = trace_dir / "combined_23.png"

    valued = load_json(valued_path)
    crosscheck = load_json(crosscheck_path)
    geojson = load_json(geojson_path)

    features = geojson.get("features", [])
    kinds = Counter(item.get("properties", {}).get("kind", "unknown") for item in features)
    sources = Counter(item.get("properties", {}).get("source", "unknown") for item in features)
    verdicts = Counter(crosscheck.get("verdicts", {}).values())
    candidate_count = int(valued.get("n_polylines", len(features)))
    profile_leaks = len(valued.get("profile_leaks", []))
    isoline_count = int(kinds.get("isoline", max(0, candidate_count - profile_leaks)))
    directly_valued = int(valued.get("n_valued", 0))
    surface_assigned = len(crosscheck.get("from_surface", {}))
    compared = verdicts["main"] + verdicts["flagged"]
    within_one_step = verdicts["main"]
    disagreements = verdicts["flagged"]
    closures = sum(bool(item.get("properties", {}).get("closed")) for item in features)
    crossing_pairs = crossing_isoline_pairs(features)

    reasons = []
    if ratio(profile_leaks, candidate_count) > 0.1:
        reasons.append("Высокая доля линий профилей среди исходных кандидатов.")
    if ratio(directly_valued, isoline_count) < 0.8:
        reasons.append("Большинство изолиний не имеет значения, прочитанного непосредственно с подписи.")
    if disagreements:
        reasons.append("Есть расхождения между подписью линии и поверхностью по отметкам карты.")
    if not closures:
        reasons.append("Замкнутые контуры не подтверждены, структуры автоматически не выделены.")
    if crossing_pairs:
        reasons.append(f"Обнаружены топологические пересечения изогипс: {len(crossing_pairs)} пар.")

    metrics = {
        "quality_status": "review" if reasons else "pass",
        "coordinate_system": "image_pixels",
        "georeferenced": False,
        "candidate_polylines": candidate_count,
        "isoline_features": isoline_count,
        "profile_leaks": profile_leaks,
        "profile_leak_rate": ratio(profile_leaks, candidate_count),
        "directly_valued": directly_valued,
        "direct_value_rate": ratio(directly_valued, isoline_count),
        "surface_assigned": surface_assigned,
        "valued_total": sum(
            item.get("properties", {}).get("value_km") is not None for item in features
        ),
        "crosscheck_within_one_step": within_one_step,
        "crosscheck_compared": compared,
        "crosscheck_within_one_step_rate": ratio(within_one_step, compared),
        "crosscheck_disagreements": disagreements,
        "closed_contours": closures,
        "crossing_isoline_pairs": len(crossing_pairs),
        "crossing_pair_ids": crossing_pairs,
        "structures": len(crosscheck.get("structures", [])),
        "contour_interval_km": valued.get("contour_step_km"),
        "source_breakdown": dict(sources),
    }

    artifacts = [
        {
            "name": "trace_preview",
            "label": "Изолинии и контрольная поверхность, лист 23",
            "path": str(preview_path.resolve()),
            "media_type": "image/png",
        },
        {
            "name": "pixel_isolines",
            "label": "Векторные изолинии в пикселях (без геопривязки)",
            "path": str(geojson_path.resolve()),
            "media_type": "application/geo+json",
        },
        {
            "name": "trace_qc",
            "label": "Метрики контроля трассировки",
            "path": str(crosscheck_path.resolve()),
            "media_type": "application/json",
        },
    ]
    if source_image and source_image.exists():
        artifacts.insert(
            0,
            {
                "name": "source_map",
                "label": "Исходная карта, лист 23",
                "path": str(source_image.resolve()),
                "media_type": "image/jpeg",
            },
        )

    payload = {
        "schema_version": 1,
        "report_id": report_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "quality_status": metrics["quality_status"],
        "artifacts": artifacts,
        "metrics": metrics,
        "issues": [
            {
                "code": "map_trace_review",
                "message": " ".join(reasons),
                "severity": "review",
            }
        ]
        if reasons
        else [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Adopt traced isolines into the map-agent contract")
    parser.add_argument("report_id")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-image", type=Path)
    args = parser.parse_args()
    payload = build_map_result(args.report_id, args.trace_dir, args.output, args.source_image)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
