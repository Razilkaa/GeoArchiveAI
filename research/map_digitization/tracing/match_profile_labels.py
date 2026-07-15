"""Attach OCR profile numbers to detected pixel lines and inventory profiles."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np


PROFILE_NUMBER = re.compile(r"^\d{5,6}$")


def point_segment_distance(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    delta = p1 - p0
    if np.dot(delta, delta) == 0:
        return float(np.linalg.norm(point - p0))
    fraction = np.clip(np.dot(point - p0, delta) / np.dot(delta, delta), 0.0, 1.0)
    return float(np.linalg.norm(point - (p0 + fraction * delta)))


def quad_center(quad: list[list[float]]) -> np.ndarray:
    return np.asarray(quad, dtype=float).mean(axis=0)


def load_labels(readings_path: Path) -> list[dict]:
    labels = []
    for raw_line in readings_path.read_text(encoding="utf-8").splitlines():
        reading = json.loads(raw_line)
        if reading.get("zone") != "map_body":
            continue
        for value in reading.get("values") or []:
            text = str(value.get("text", "")).strip().replace(" ", "")
            if not PROFILE_NUMBER.fullmatch(text):
                continue
            labels.append(
                {
                    "reading_id": reading["id"],
                    "text": text,
                    "center": quad_center(reading["quad"]).round(2).tolist(),
                    "text_angle_deg": reading.get("angle"),
                    "ocr_confidence": value.get("confidence"),
                }
            )
    return labels


def attach_lines(labels: list[dict], lines: list[dict], max_distance: float = 60.0) -> None:
    for label in labels:
        point = np.asarray(label["center"])
        ranked = sorted(
            (
                point_segment_distance(point, np.asarray(line["p0"]), np.asarray(line["p1"])),
                line,
            )
            for line in lines
        )
        distance, line = ranked[0]
        label["line_id"] = line["id"] if distance <= max_distance else None
        label["line_distance_px"] = round(distance, 2)
        label["line_angle_deg"] = line["angle_deg"] if distance <= max_distance else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    lines_payload = json.loads(args.lines.read_text(encoding="utf-8"))
    lines = lines_payload["profile_candidates"]
    labels = load_labels(args.readings)
    attach_lines(labels, lines)

    inventory = gpd.read_file(args.inventory)
    profile_ids = {str(value).strip() for value in inventory["N_PROF"].dropna()}
    for label in labels:
        label["inventory_match"] = label["text"] if label["text"] in profile_ids else None
        if label["line_id"] is None:
            label["status"] = "no_pixel_line"
        elif label["inventory_match"] is None:
            label["status"] = "map_only"
        else:
            label["status"] = "exact_anchor"

    exact = [label for label in labels if label["status"] == "exact_anchor"]
    exact_angles = [label["line_angle_deg"] for label in exact]
    angle_spread = max(exact_angles) - min(exact_angles) if exact_angles else 0.0
    georeference_ready = len(exact) >= 3 and angle_spread >= 10.0
    payload = {
        "source_readings": str(args.readings),
        "source_lines": str(args.lines),
        "inventory": str(args.inventory),
        "inventory_crs": str(inventory.crs),
        "labels_found": len(labels),
        "exact_anchors": len(exact),
        "exact_anchor_angle_spread_deg": round(angle_spread, 2),
        "georeference_ready": georeference_ready,
        "blocker": None
        if georeference_ready
        else "Need at least three exact anchors spanning two profile directions",
        "labels": labels,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.preview:
        image_path = Path(lines_payload["source"])
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        colors = {"exact_anchor": (0, 150, 0), "map_only": (0, 140, 255), "no_pixel_line": (0, 0, 255)}
        for label in labels:
            center = tuple(round(value) for value in label["center"])
            color = colors[label["status"]]
            cv2.circle(image, center, 24, color, 6, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{label['text']} [{label['line_id']}]",
                (center[0] + 20, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                color,
                3,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(args.preview), image)

    print(
        f"{len(labels)} labels, {len(exact)} exact inventory anchors; "
        f"georeference_ready={georeference_ready}"
    )


if __name__ == "__main__":
    main()
