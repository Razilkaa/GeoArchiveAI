"""Resumable sequential runner for large map collections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from services.map_digitizer.benchmark import summarize_results
from services.map_digitizer import PIPELINE_VERSION
from services.map_digitizer.pipeline import file_sha256, run_pipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TERMINAL_STATUSES = {"accepted", "review", "not_applicable"}


def discover_images(inputs: list[Path]) -> list[Path]:
    images = []
    for source in inputs:
        if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source.resolve())
        elif source.is_dir():
            images.extend(
                path.resolve()
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    return sorted(set(images), key=lambda path: str(path).lower())


def case_directory(output_dir: Path, image_path: Path) -> Path:
    path_key = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(character if character.isalnum() else "_" for character in image_path.stem)
    return output_dir / f"{safe_stem[:48]}_{path_key}"


def reusable_result(path: Path, source: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in TERMINAL_STATUSES:
        return None
    if payload.get("version") != PIPELINE_VERSION:
        return None
    if Path(payload.get("source", "")).resolve() != source.resolve():
        return None
    source_stat = source.stat()
    if (
        payload.get("source_size_bytes") == source_stat.st_size
        and payload.get("source_mtime_ns") == source_stat.st_mtime_ns
    ):
        return payload
    expected_hash = payload.get("source_sha256")
    if expected_hash and expected_hash != file_sha256(source):
        return None
    return payload


def run_batch(
    images: list[Path],
    output_dir: Path,
    *,
    ocr_api_url: str = "http://127.0.0.1:18080",
    trace_scale: float = 0.6,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "batch_result.json"
    results = []
    if not images:
        manifest = {
            "version": 1,
            "processed": 0,
            "total": 0,
            "last_reused": False,
            "summary": summarize_results([]),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
    for index, image_path in enumerate(images, start=1):
        case_dir = case_directory(output_dir, image_path)
        result_path = case_dir / "pipeline_result.json"
        payload = reusable_result(result_path, image_path)
        reused = payload is not None
        if payload is None:
            try:
                payload = run_pipeline(
                    image_path,
                    case_dir,
                    ocr_api_url=ocr_api_url,
                    trace_scale=trace_scale,
                )
            except Exception:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(payload)
        manifest = {
            "version": 1,
            "processed": index,
            "total": len(images),
            "last_reused": reused,
            "summary": summarize_results(results),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(images),
                    "source": str(image_path),
                    "status": payload.get("status"),
                    "reused": reused,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Digitize image files sequentially and resumably")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ocr-api-url", default="http://127.0.0.1:18080")
    parser.add_argument("--trace-scale", type=float, default=0.6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    images = discover_images(args.inputs)
    if args.limit is not None:
        images = images[: max(0, args.limit)]
    result = run_batch(
        images,
        args.output_dir,
        ocr_api_url=args.ocr_api_url,
        trace_scale=args.trace_scale,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
