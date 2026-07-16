from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import BaseModel


MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(40 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "16"))
DEVICE = os.getenv("OCR_DEVICE", "gpu:0")
LANGUAGE = os.getenv("OCR_LANGUAGE", "ru")
MODEL_NAME = os.getenv("OCR_MODEL", "PP-OCRv5")
AUTOCONTRAST_RETRY = os.getenv("OCR_AUTOCONTRAST_RETRY", "true").lower() not in {"0", "false", "no"}
RETRY_MIN_LINES = max(0, int(os.getenv("OCR_RETRY_MIN_LINES", "3")))
RETRY_MIN_CONFIDENCE = float(os.getenv("OCR_RETRY_MIN_CONFIDENCE", "0.93"))
logger = logging.getLogger("geoarchive.ocr")


class ServiceState:
    model: Any = None
    semaphore: asyncio.Semaphore
    started_at: float
    requests: int = 0
    pages: int = 0
    errors: int = 0
    total_latency_s: float = 0.0


state = ServiceState()


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _normalize_result(result: Any) -> dict[str, Any]:
    payload = _result_payload(result)
    texts = list(payload.get("rec_texts") or [])
    scores = list(payload.get("rec_scores") or [])
    polygons = list(payload.get("rec_polys") or payload.get("dt_polys") or [])
    lines = []
    for index, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        polygon = polygons[index] if index < len(polygons) else None
        if hasattr(polygon, "tolist"):
            polygon = polygon.tolist()
        score = float(scores[index]) if index < len(scores) else None
        lines.append({"text": text, "score": score, "polygon": polygon})
    return {
        "text": "\n".join(item["text"] for item in lines),
        "lines": lines,
        "line_count": len(lines),
    }


def _predict(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    results = list(state.model.predict(str(path)))
    normalized = [_normalize_result(result) for result in results]
    lines = [line for page in normalized for line in page["lines"]]
    return {
        "engine": "paddleocr",
        "model": MODEL_NAME,
        "language": LANGUAGE,
        "device": DEVICE,
        "latency_s": round(time.perf_counter() - started, 3),
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "line_count": len(lines),
    }


async def _read_upload(upload: UploadFile) -> bytes:
    payload = await upload.read(MAX_FILE_BYTES + 1)
    if not payload:
        raise HTTPException(400, "Empty file")
    if len(payload) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_FILE_BYTES} bytes")
    return payload


def _write_normalized_image(payload: bytes, destination: Any, *, enhanced: bool = False) -> None:
    """Decode scans and normalize faded paper before Paddle text detection."""
    with Image.open(io.BytesIO(payload)) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        if enhanced:
            image = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)
            image = ImageEnhance.Contrast(image).enhance(1.35)
            image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        image = image.convert("RGB")
        destination.seek(0)
        destination.truncate()
        image.save(destination, format="JPEG", quality=95, optimize=True)


def _mean_confidence(result: dict[str, Any]) -> float:
    scores = [line["score"] for line in result["lines"] if line.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0.0


async def _ocr_upload(upload: UploadFile) -> dict[str, Any]:
    payload = await _read_upload(upload)
    async with state.semaphore:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            try:
                _write_normalized_image(payload, handle)
            except (OSError, UnidentifiedImageError) as error:
                state.errors += 1
                raise HTTPException(422, f"Unsupported image: {type(error).__name__}") from error
            handle.flush()
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(_predict, Path(handle.name))
                result["preprocessing"] = "original"
                retry = result["line_count"] < RETRY_MIN_LINES or _mean_confidence(result) < RETRY_MIN_CONFIDENCE
                if AUTOCONTRAST_RETRY and retry:
                    _write_normalized_image(payload, handle, enhanced=True)
                    handle.flush()
                    enhanced = await asyncio.to_thread(_predict, Path(handle.name))
                    if _mean_confidence(enhanced) > _mean_confidence(result):
                        result = enhanced
                        result["preprocessing"] = "enhanced_retry"
            except Exception as error:
                state.errors += 1
                logger.exception("OCR prediction failed for %s", upload.filename)
                raise HTTPException(500, f"OCR failed: {type(error).__name__}: {error}") from error
            state.requests += 1
            state.pages += 1
            elapsed = time.perf_counter() - started
            state.total_latency_s += elapsed
            result["latency_s"] = round(elapsed, 3)
            result["filename"] = upload.filename
            return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    from paddleocr import PaddleOCR

    state.started_at = time.time()
    state.semaphore = asyncio.Semaphore(max(1, int(os.getenv("OCR_CONCURRENCY", "1"))))
    state.model = PaddleOCR(
        lang=LANGUAGE,
        ocr_version=MODEL_NAME,
        device=DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    yield
    state.model = None


app = FastAPI(title="GeoArchive PaddleOCR", version="1.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.model is not None else "starting",
        "model": MODEL_NAME,
        "language": LANGUAGE,
        "device": DEVICE,
        "autocontrast_retry": AUTOCONTRAST_RETRY,
        "retry_min_lines": RETRY_MIN_LINES,
        "retry_min_confidence": RETRY_MIN_CONFIDENCE,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "requests": state.requests,
        "pages": state.pages,
        "errors": state.errors,
        "average_latency_s": round(state.total_latency_s / state.pages, 3) if state.pages else 0.0,
        "uptime_s": round(time.time() - state.started_at, 1),
    }


@app.post("/v1/ocr")
async def ocr(file: UploadFile = File(...)) -> dict[str, Any]:
    return await _ocr_upload(file)


@app.post("/v1/ocr:batch")
async def ocr_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"Batch exceeds {MAX_BATCH_FILES} files")
    started = time.perf_counter()
    pages = [await _ocr_upload(file) for file in files]
    return {"pages": pages, "page_count": len(pages), "latency_s": round(time.perf_counter() - started, 3)}
