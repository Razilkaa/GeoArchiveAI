"""Memory-bounded OCR client for large archival scans."""
from __future__ import annotations

import math
import mimetypes
import tempfile
import time
from pathlib import Path
from typing import Callable

import requests
from PIL import Image


MAX_DIRECT_PIXELS = 40_000_000
TILE_SIZE = 4000


def _post_ocr(image_path: Path, api_url: str, timeout: float) -> dict:
    session = requests.Session()
    session.trust_env = False
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.open("rb") as source:
        response = session.post(
            f"{api_url.rstrip('/')}/v1/ocr",
            files={"file": (image_path.name, source, mime_type)},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("lines"), list):
        raise ValueError("OCR response has no lines array")
    return payload


def tile_starts(length: int, tile_size: int = TILE_SIZE) -> list[int]:
    if length <= tile_size:
        return [0]
    count = math.ceil(length / tile_size)
    return [round(index * (length - tile_size) / (count - 1)) for index in range(count)]


def _offset_line(line: dict, offset_x: int, offset_y: int) -> dict:
    shifted = dict(line)
    polygon = line.get("polygon") or line.get("points")
    if polygon:
        shifted["polygon"] = [
            [float(point[0]) + offset_x, float(point[1]) + offset_y] for point in polygon
        ]
        shifted.pop("points", None)
    return shifted


def _scale_line(line: dict, scale_x: float, scale_y: float) -> dict:
    scaled = dict(line)
    polygon = line.get("polygon") or line.get("points")
    if polygon:
        scaled["polygon"] = [
            [float(point[0]) * scale_x, float(point[1]) * scale_y]
            for point in polygon
        ]
        scaled.pop("points", None)
    return scaled


def _bounds(line: dict) -> tuple[float, float, float, float] | None:
    polygon = line.get("polygon") or line.get("points")
    if not polygon:
        return None
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _intersection_over_minimum(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def deduplicate_lines(lines: list[dict], overlap_threshold: float = 0.45) -> list[dict]:
    accepted: list[dict] = []
    bounds: list[tuple[float, float, float, float] | None] = []
    for candidate in sorted(lines, key=lambda line: float(line.get("score") or 0), reverse=True):
        candidate_bounds = _bounds(candidate)
        duplicate = False
        if candidate_bounds:
            for existing_bounds in bounds:
                if existing_bounds and _intersection_over_minimum(
                    candidate_bounds, existing_bounds
                ) >= overlap_threshold:
                    duplicate = True
                    break
        if not duplicate:
            accepted.append(candidate)
            bounds.append(candidate_bounds)
    accepted.sort(key=lambda line: (_bounds(line) or (0, 0, 0, 0))[1::-1])
    return accepted


def request_ocr(
    image_path: Path,
    api_url: str,
    timeout: float = 120.0,
    *,
    post: Callable[[Path, str, float], dict] = _post_ocr,
    max_direct_pixels: int = MAX_DIRECT_PIXELS,
    tile_size: int = TILE_SIZE,
) -> dict:
    with Image.open(image_path) as source:
        width, height = source.size
        if width * height <= max_direct_pixels:
            return post(image_path, api_url, timeout)

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="geoarchive_ocr_") as temporary:
            temporary_root = Path(temporary)
            resize_scale = min(1.0, tile_size / max(width, height))
            resized_width = max(1, round(width * resize_scale))
            resized_height = max(1, round(height * resize_scale))
            normalized_path = temporary_root / "normalized.png"
            source.resize(
                (resized_width, resized_height), Image.Resampling.LANCZOS
            ).save(normalized_path, format="PNG", optimize=False)
            try:
                payload = post(normalized_path, api_url, timeout)
            except requests.RequestException:
                payload = None
            if payload is not None:
                lines = [
                    _scale_line(line, width / resized_width, height / resized_height)
                    for line in payload.get("lines") or []
                ]
                return {
                    **{key: value for key, value in payload.items() if key not in {"lines", "text"}},
                    "latency_s": round(time.perf_counter() - started, 3),
                    "text": "\n".join(str(line.get("text", "")) for line in lines),
                    "lines": lines,
                    "line_count": len(lines),
                    "preprocessing": "downscaled",
                    "scale_to_source": [width / resized_width, height / resized_height],
                    "filename": image_path.name,
                    "image_size": [width, height],
                }

            all_lines: list[dict] = []
            tile_count = 0
            engine_payload: dict = {}
            for y in tile_starts(height, tile_size):
                for x in tile_starts(width, tile_size):
                    tile_count += 1
                    tile_path = temporary_root / f"tile_{y}_{x}.png"
                    source.crop((x, y, min(x + tile_size, width), min(y + tile_size, height))).save(
                        tile_path, format="PNG", optimize=False
                    )
                    payload = post(tile_path, api_url, timeout)
                    if not engine_payload:
                        engine_payload = payload
                    all_lines.extend(
                        _offset_line(line, x, y) for line in payload.get("lines") or []
                    )

    lines = deduplicate_lines(all_lines)
    return {
        "engine": engine_payload.get("engine", "paddleocr"),
        "model": engine_payload.get("model"),
        "language": engine_payload.get("language"),
        "device": engine_payload.get("device"),
        "latency_s": round(time.perf_counter() - started, 3),
        "text": "\n".join(str(line.get("text", "")) for line in lines),
        "lines": lines,
        "line_count": len(lines),
        "preprocessing": "tiled",
        "tile_count": tile_count,
        "tile_size": tile_size,
        "filename": image_path.name,
        "image_size": [width, height],
    }
