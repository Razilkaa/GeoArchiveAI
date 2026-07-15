from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import cv2
import fitz
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps

from factory_runner import load_manifest, read_credentials, save_manifest, set_stage
from report_factory import choose_fast_ocr, classify_path


KINDS = {
    "text",
    "toc",
    "map",
    "seismic_section",
    "profile_scheme",
    "chart",
    "table",
    "photo",
    "other_graphic",
    "blank",
}
GRAPHIC_KINDS = {"map", "seismic_section", "profile_scheme", "chart", "other_graphic"}
ROUTER_VERSION = 2

SYSTEM_PROMPT = """Classify scanned pages of a geological report from one contact sheet.
Return only JSON: {"pages":[{"index":0,"kind":"text","confidence":0.95}]}.
Use exactly one kind: text, toc, map, seismic_section, profile_scheme, table,
chart, photo, other_graphic, blank. Use chart for sheets dominated by plotted curves or
multiple axes. Classify layout and content type; transcription is not needed.
Return every visible numeric index exactly once."""


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheet(
    pages: list[dict[str, Any]],
    source_root: Path,
    *,
    cell_size: tuple[int, int] = (420, 320),
    columns: int = 4,
) -> Image.Image:
    if not pages:
        raise ValueError("Cannot build an empty contact sheet")
    width, height = cell_size
    rows = math.ceil(len(pages) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    font = _font(24)
    pdf_documents: dict[Path, fitz.Document] = {}
    try:
        for index, page in enumerate(pages):
            try:
                preview = load_page_preview(page, source_root, (width - 12, height - 42), pdf_documents)
            except (OSError, ValueError, RuntimeError):
                preview = Image.new("RGB", (width - 12, height - 42), "#f3f3f3")
                placeholder = ImageDraw.Draw(preview)
                placeholder.line((20, 20, preview.width - 20, preview.height - 20), fill="#aa0000", width=4)
                placeholder.line((preview.width - 20, 20, 20, preview.height - 20), fill="#aa0000", width=4)
            tile = Image.new("RGB", cell_size, "white")
            x = (width - preview.width) // 2
            y = 36 + (height - 36 - preview.height) // 2
            tile.paste(preview, (x, y))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, width - 1, height - 1), outline="#555555", width=2)
            draw.rectangle((0, 0, 76, 34), fill="white", outline="#222222", width=1)
            draw.text((8, 3), str(index), fill="black", font=font)
            sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    finally:
        for document in pdf_documents.values():
            document.close()
    return sheet


def load_page_preview(
    page: dict[str, Any],
    source_root: Path,
    target: tuple[int, int],
    pdf_documents: dict[Path, fitz.Document] | None = None,
) -> Image.Image:
    source = source_root / Path(page["relative_path"])
    if page.get("source_kind") != "pdf":
        return load_preview(source, target)
    documents = pdf_documents if pdf_documents is not None else {}
    document = documents.get(source)
    owned = document is None and pdf_documents is None
    if document is None:
        document = fitz.open(source)
        documents[source] = document
    try:
        pdf_page = document.load_page(int(page["pdf_page"]))
        scale = min(target[0] / pdf_page.rect.width, target[1] / pdf_page.rect.height)
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        preview = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        preview.thumbnail(target, Image.Resampling.LANCZOS)
        return preview
    finally:
        if owned:
            document.close()


def load_preview(source: Path, target: tuple[int, int]) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(source) as scan:
                if scan.width * scan.height <= 25_000_000:
                    preview = ImageOps.exif_transpose(scan).convert("RGB")
                    preview.thumbnail(target, Image.Resampling.LANCZOS)
                    return preview.copy()
    except Image.DecompressionBombError:
        pass
    encoded = np.fromfile(source, dtype=np.uint8)
    reduced = cv2.imdecode(encoded, cv2.IMREAD_REDUCED_COLOR_8)
    if reduced is None:
        raise ValueError(f"Cannot decode oversized scan: {source}")
    preview = Image.fromarray(cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB))
    preview.thumbnail(target, Image.Resampling.LANCZOS)
    return preview


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + __import__("base64").b64encode(buffer.getvalue()).decode("ascii")


def parse_decisions(text: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    records = payload.get("pages", [])
    by_index = {int(item["index"]): item for item in records if "index" in item}
    decisions = []
    for index, page in enumerate(batch):
        item = by_index.get(index)
        if item is None:
            raise ValueError(f"Router omitted page index {index}")
        kind = str(item.get("kind") or "").strip()
        if kind not in KINDS:
            raise ValueError(f"Unsupported page kind: {kind}")
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        decisions.append({"page_id": page["id"], "kind": kind, "confidence": confidence})
    return decisions


def classify_contact_sheet(
    batch: list[dict[str, Any]], source_root: Path, client: OpenAI, model: str
) -> list[dict[str, Any]]:
    sheet = build_contact_sheet(batch, source_root)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": _data_url(sheet), "detail": "high"}},
                ],
            }
        ],
    )
    return parse_decisions(response.choices[0].message.content or "{}", batch)


def apply_decisions(manifest: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    by_id = {item["page_id"]: item for item in decisions}
    source_root = Path(manifest.get("source_root") or ".")
    for page in manifest["pages"]:
        decision = by_id.get(page["id"])
        if not decision:
            continue
        kind = decision["kind"]
        source_path = source_root / Path(page["relative_path"])
        path_role, _ = classify_path(source_path, source_root)
        original_role = path_role if path_role != "unknown" else (page.get("role") or "unknown")
        page["content_type"] = kind
        page["routing_confidence"] = decision["confidence"]
        page["role_source"] = "path+preview_vlm" if original_role != "unknown" else "preview_vlm"
        page["route_vlm"] = False
        page["fast_ocr"] = False
        page["fast_ocr_reasons"] = []
        if original_role == "toc":
            page["role"] = "toc"
        elif original_role == "text":
            page["role"] = "text"
        elif original_role in {"graphic", "sample"}:
            page["role"] = "graphic"
        elif kind == "toc":
            page["role"] = "toc"
        elif kind in {"text", "table"}:
            page["role"] = "text"
        elif kind in GRAPHIC_KINDS or kind == "photo":
            page["role"] = "graphic"
        else:
            page["role"] = "unknown"
        if original_role == "graphic":
            page["deep_processing"] = True
        elif original_role in {"toc", "text", "sample"}:
            page["deep_processing"] = False
        else:
            page["deep_processing"] = kind in GRAPHIC_KINDS

    choose_fast_ocr(manifest["pages"])
    manifest["queues"]["fast_ocr"] = [page["id"] for page in manifest["pages"] if page["fast_ocr"]]
    manifest["queues"]["route_vlm"] = [page["id"] for page in manifest["pages"] if page["route_vlm"]]
    manifest["queues"]["deep_processing"] = [
        page["id"] for page in manifest["pages"] if page["deep_processing"]
    ]
    manifest["summary"]["fast_ocr_pages"] = len(manifest["queues"]["fast_ocr"])
    manifest["summary"]["route_vlm_pages"] = len(manifest["queues"]["route_vlm"])
    manifest["summary"]["deep_processing_pages"] = len(manifest["queues"]["deep_processing"])
    manifest["summary"]["roles"] = dict(sorted(Counter(page["role"] for page in manifest["pages"]).items()))


def restore_explicit_path_roles(manifest: dict[str, Any]) -> None:
    source_root = Path(manifest.get("source_root") or ".")
    for page in manifest["pages"]:
        path_role, _ = classify_path(source_root / Path(page["relative_path"]), source_root)
        if path_role == "unknown":
            continue
        page["role"] = path_role
        page["role_source"] = "path"
        if path_role in {"graphic", "sample"}:
            page["fast_ocr"] = False
            page["fast_ocr_reasons"] = []
            page["deep_processing"] = path_role == "graphic"


def route_pages(
    manifest_path: Path,
    credentials_file: Path,
    *,
    model: str = "openai/gpt-4o-mini",
    batch_size: int = 12,
    workers: int = 4,
    route_all: bool = False,
    classifier: Callable[[list[dict[str, Any]], Path, OpenAI, str], list[dict[str, Any]]] = classify_contact_sheet,
) -> Path:
    manifest = load_manifest(manifest_path)
    restore_explicit_path_roles(manifest)
    apply_decisions(manifest, [])
    save_manifest(manifest_path, manifest)
    source_root = Path(manifest["source_root"])
    pages = manifest["pages"] if route_all else [page for page in manifest["pages"] if page["route_vlm"]]
    output = manifest_path.parent / "page_routing.json"
    if not pages:
        cached_payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"pages": []}
        cached_pages = cached_payload.get("pages", [])
        if not cached_pages:
            cache_dir = manifest_path.parent / "page_routing_cache"
            by_page_id = {}
            for cache_path in sorted(cache_dir.glob("batch_*.json")):
                for item in json.loads(cache_path.read_text(encoding="utf-8")).get("pages", []):
                    by_page_id[item["page_id"]] = item
            cached_pages = list(by_page_id.values())
        if cached_pages:
            apply_decisions(manifest, cached_pages)
            save_manifest(manifest_path, manifest)
        payload = {
            "report_id": manifest["report_id"],
            "model": model,
            "routed_pages": len(cached_pages),
            "requests": 0,
            "elapsed_s": 0,
            "cache_hit": True,
            "pages": cached_pages,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        set_stage(
            manifest_path,
            "page_routing",
            "completed",
            routed_pages=len(cached_pages),
            requests=0,
            elapsed_s=0,
            cache_hit=True,
        )
        return output

    credentials = read_credentials(credentials_file)
    client = OpenAI(
        api_key=credentials["OPENAI_API_KEY"],
        base_url=credentials.get("BASE_URL"),
        timeout=120,
    )
    cache_dir = manifest_path.parent / "page_routing_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_stage(manifest_path, "page_routing", "running", routed_pages=len(pages), model=model)
    started = time.perf_counter()
    decisions_by_batch: dict[int, list[dict[str, Any]]] = {}
    pending_batches = []
    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        signature = "\n".join(
            f"{page['id']}:{page.get('size_bytes')}:{page.get('mtime_ns')}" for page in batch
        )
        key = hashlib.sha256(f"router:{ROUTER_VERSION}\n{model}\n{signature}".encode("utf-8")).hexdigest()[:20]
        cache_path = cache_dir / f"batch_{key}.json"
        if cache_path.exists():
            decisions_by_batch[start] = json.loads(cache_path.read_text(encoding="utf-8"))["pages"]
        else:
            pending_batches.append((start, batch, cache_path))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending_batches) or 1))) as pool:
        futures = {
            pool.submit(classifier, batch, source_root, client, model): (start, cache_path)
            for start, batch, cache_path in pending_batches
        }
        for future in as_completed(futures):
            start, cache_path = futures[future]
            batch_decisions = future.result()
            cache_path.write_text(
                json.dumps({"pages": batch_decisions}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            decisions_by_batch[start] = batch_decisions
    requests = len(pending_batches)
    decisions = [item for start in sorted(decisions_by_batch) for item in decisions_by_batch[start]]

    apply_decisions(manifest, decisions)
    save_manifest(manifest_path, manifest)
    elapsed = round(time.perf_counter() - started, 2)
    payload = {
        "report_id": manifest["report_id"],
        "model": model,
        "routed_pages": len(decisions),
        "requests": requests,
        "elapsed_s": elapsed,
        "pages": decisions,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    set_stage(
        manifest_path,
        "page_routing",
        "completed",
        routed_pages=len(decisions),
        requests=requests,
        elapsed_s=elapsed,
        model=model,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Route report pages from batched low-resolution previews")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--all", action="store_true", dest="route_all")
    args = parser.parse_args()
    print(
        route_pages(
            args.manifest,
            args.credentials_file,
            model=args.model,
            batch_size=args.batch_size,
            workers=args.workers,
            route_all=args.route_all,
        )
    )


if __name__ == "__main__":
    main()
