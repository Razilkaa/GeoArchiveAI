from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}
STAGES = (
    "inventory",
    "page_routing",
    "fast_ocr",
    "ragflow_ingest",
    "entity_agents",
    "result_bundle",
    "deep_processing",
)


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(path))]


def page_number(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def classify_path(path: Path, report_root: Path) -> tuple[str, str | None]:
    relative = path.relative_to(report_root)
    parts = [part.casefold() for part in relative.parts[:-1]]
    volume = next(
        (part for part in relative.parts[:-1] if re.match(r"^[tт]ом\b", part, re.IGNORECASE)),
        None,
    )
    if any("оглав" in part for part in parts):
        return "toc", volume
    if any("выборк" in part for part in parts):
        return "sample", volume
    if any(part.endswith("екст") for part in parts):
        return "text", volume
    if any("график" in part for part in parts):
        return "graphic", volume
    return "unknown", volume


def discover_pages(report_root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (
            path
            for path in report_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES | PDF_SUFFIXES
        ),
        key=natural_key,
    )
    pages = []
    for path in paths:
        role, volume = classify_path(path, report_root)
        stat = path.stat()
        if path.suffix.casefold() in PDF_SUFFIXES:
            with fitz.open(path) as document:
                page_records = [(page_index, page_index + 1) for page_index in range(document.page_count)]
            if volume is None and re.match(r"^[tт]ом\b", path.stem, re.IGNORECASE):
                volume = path.stem
        else:
            page_records = [(None, page_number(path))]
        for pdf_page, logical_page in page_records:
            index = len(pages)
            record = {
                "id": f"page:{index:05d}",
                "relative_path": path.relative_to(report_root).as_posix(),
                "source_kind": "pdf" if pdf_page is not None else "image",
                "role": role,
                "volume": volume,
                "page_number": logical_page,
                "size_bytes": stat.st_size if pdf_page is None else max(1, stat.st_size // len(page_records)),
                "mtime_ns": stat.st_mtime_ns,
                "fast_ocr": False,
                "fast_ocr_reasons": [],
                "route_vlm": role in {"graphic", "toc", "unknown"},
                "deep_processing": role == "graphic",
            }
            if pdf_page is not None:
                record["pdf_page"] = pdf_page
            pages.append(record)
    return pages


def choose_fast_ocr(pages: list[dict[str, Any]]) -> None:
    for page in pages:
        if page["role"] == "text":
            page["fast_ocr"] = True
            if "full_text_corpus" not in page["fast_ocr_reasons"]:
                page["fast_ocr_reasons"].append("full_text_corpus")
        elif page["role"] == "toc":
            page["fast_ocr"] = True
            if "explicit_toc" not in page["fast_ocr_reasons"]:
                page["fast_ocr_reasons"].append("explicit_toc")


def ensure_full_text_strategy(manifest: dict[str, Any]) -> bool:
    before = set(manifest.get("queues", {}).get("fast_ocr", []))
    previous_strategy = manifest.get("strategy")
    choose_fast_ocr(manifest["pages"])
    selected = [page["id"] for page in manifest["pages"] if page["fast_ocr"]]
    manifest.setdefault("queues", {})["fast_ocr"] = selected
    manifest.setdefault("summary", {})["fast_ocr_pages"] = len(selected)
    manifest["strategy"] = "full_text_fast_path"
    return before != set(selected) or previous_strategy != manifest["strategy"]


def build_manifest(report_root: Path, report_id: str | None = None) -> dict[str, Any]:
    report_root = report_root.resolve()
    if not report_root.is_dir():
        raise ValueError(f"Report directory does not exist: {report_root}")
    pages = discover_pages(report_root)
    if not pages:
        raise ValueError(f"No scan images found in {report_root}")
    choose_fast_ocr(pages)
    counts = Counter(page["role"] for page in pages)
    fast_pages = [page for page in pages if page["fast_ocr"]]
    route_pages = [page for page in pages if page["route_vlm"]]
    return {
        "schema_version": 1,
        "report_id": report_id or report_root.name,
        "source_root": str(report_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "full_text_fast_path",
        "targets": ["structures", "wells", "maps", "horizons", "key_results"],
        "pages": pages,
        "queues": {
            "fast_ocr": [page["id"] for page in fast_pages],
            "route_vlm": [page["id"] for page in route_pages],
            "deep_processing": [page["id"] for page in pages if page["deep_processing"]],
        },
        "summary": {
            "page_count": len(pages),
            "roles": dict(sorted(counts.items())),
            "fast_ocr_pages": len(fast_pages),
            "route_vlm_pages": len(route_pages),
            "deep_processing_pages": sum(page["deep_processing"] for page in pages),
        },
        "stages": {
            stage: {
                "status": "completed" if stage == "inventory" else "pending",
                "started_at": None,
                "finished_at": None,
                "metrics": {},
                "error": None,
            }
            for stage in STAGES
        },
    }


def write_manifest(manifest: dict[str, Any], runs_root: Path) -> Path:
    output_dir = runs_root / manifest["report_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "job.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a resumable fast-path report processing job")
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--report-id")
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).parents[1] / "runs")
    args = parser.parse_args()
    manifest = build_manifest(args.report_root, args.report_id)
    output = write_manifest(manifest, args.runs_root)
    print(json.dumps({"manifest": str(output), **manifest["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
