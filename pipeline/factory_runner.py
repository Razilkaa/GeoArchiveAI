from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import urllib3
import fitz
from openai import OpenAI
from PIL import Image


urllib3.disable_warnings()


def read_token(path: Path) -> str:
    line = next(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return line.split("=", 1)[-1].strip().strip('"').strip("'")


def read_credentials(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def set_stage(path: Path, stage: str, status: str, **metrics: Any) -> None:
    manifest = load_manifest(path)
    record = manifest["stages"][stage]
    record["status"] = status
    if status == "running" and not record.get("started_at"):
        record["started_at"] = time.time()
    if status in {"completed", "failed"}:
        record["finished_at"] = time.time()
    record["metrics"].update(metrics)
    save_manifest(path, manifest)


def selected_pages(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = set(manifest["queues"]["fast_ocr"])
    return [page for page in manifest["pages"] if page["id"] in selected]


def prepare_ocr_queue(manifest_path: Path) -> Path:
    manifest = load_manifest(manifest_path)
    source_root = Path(manifest["source_root"])
    queue_dir = manifest_path.parent / "fast_ocr" / "input"
    queue_dir.mkdir(parents=True, exist_ok=True)
    records = []
    pdf_documents: dict[Path, fitz.Document] = {}
    try:
        for page in selected_pages(manifest):
            source = source_root / Path(page["relative_path"])
            suffix = ".jpg" if page.get("source_kind") == "pdf" else (source.suffix.casefold() or ".jpg")
            destination = queue_dir / f"{page['id'].replace(':', '_')}{suffix}"
            if page.get("source_kind") == "pdf":
                if not destination.exists():
                    document = pdf_documents.get(source)
                    if document is None:
                        document = fitz.open(source)
                        pdf_documents[source] = document
                    pdf_page = document.load_page(int(page["pdf_page"]))
                    scale = min(200 / 72, 3000 / max(pdf_page.rect.width, pdf_page.rect.height))
                    pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    rendered = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    rendered.save(destination, format="JPEG", quality=90, optimize=True)
            elif not destination.exists() or destination.stat().st_size != source.stat().st_size:
                if destination.exists():
                    destination.unlink()
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            record = {
                "page_id": page["id"],
                "input_name": destination.name,
                "source_path": page["relative_path"],
                "reasons": page["fast_ocr_reasons"],
            }
            if page.get("source_kind") == "pdf":
                record["pdf_page"] = page["pdf_page"]
            records.append(record)
    finally:
        for document in pdf_documents.values():
            document.close()
    queue_path = manifest_path.parent / "fast_ocr" / "queue.json"
    queue_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    set_stage(manifest_path, "fast_ocr", "pending", queued_pages=len(records))
    return queue_path


def _run_ocr_worker(paddleocr: Path, inputs: list[Path], worker_dir: Path) -> dict[str, Any]:
    input_dir = worker_dir / "input"
    output_dir = worker_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in inputs:
        destination = input_dir / source.name
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    command = [
        str(paddleocr), "ocr", "-i", str(input_dir), "--save_path", str(output_dir),
        "--text_detection_model_name", "PP-OCRv5_mobile_det",
        "--text_recognition_model_name", "eslav_PP-OCRv5_mobile_rec",
        "--use_doc_orientation_classify", "False", "--use_doc_unwarping", "False",
        "--use_textline_orientation", "False", "--text_det_limit_side_len", "2400",
        "--device", "cpu", "--cpu_threads", "2", "--enable_mkldnn", "False",
    ]
    log_path = worker_dir / "paddleocr.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {"worker": worker_dir.name, "return_code": completed.returncode, "pages": len(inputs)}


def run_fast_ocr(manifest_path: Path, paddleocr: Path, workers: int = 4) -> Path:
    queue_path = prepare_ocr_queue(manifest_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    input_dir = queue_path.parent / "input"
    output_dir = queue_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for item in queue:
        source = input_dir / item["input_name"]
        expected = output_dir / f"{source.stem}_res.json"
        if not expected.exists():
            pending.append(source)
    if not pending:
        set_stage(manifest_path, "fast_ocr", "completed", completed_pages=len(queue), elapsed_s=0)
        return output_dir

    set_stage(manifest_path, "fast_ocr", "running", pending_pages=len(pending), workers=workers)
    started = time.perf_counter()
    shards = [[] for _ in range(max(1, workers))]
    for index, path in enumerate(pending):
        shards[index % len(shards)].append(path)
    results = []
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futures = {
            pool.submit(_run_ocr_worker, paddleocr, shard, queue_path.parent / "shards" / f"worker_{index}"): index
            for index, shard in enumerate(shards) if shard
        }
        for future in as_completed(futures):
            results.append(future.result())
    failed = [result for result in results if result["return_code"] != 0]
    for shard_output in (queue_path.parent / "shards").glob("worker_*/output/*_res.json"):
        shutil.copy2(shard_output, output_dir / shard_output.name)
    elapsed = round(time.perf_counter() - started, 2)
    completed_count = len(list(output_dir.glob("*_res.json")))
    if failed:
        set_stage(manifest_path, "fast_ocr", "failed", elapsed_s=elapsed, failed_workers=failed)
        raise RuntimeError(f"PaddleOCR workers failed: {failed}")
    set_stage(manifest_path, "fast_ocr", "completed", elapsed_s=elapsed, completed_pages=completed_count)
    return output_dir


def _vision_data_url(path: Path, max_side: int = 2400) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _run_vision_page(path: Path, output_path: Path, client: OpenAI, model: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=3500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Распознай весь читаемый русский текст на странице архивного геологического отчета. "
                            "Сохрани порядок чтения, заголовки, номера, единицы измерения и строки таблиц. "
                            "Не объясняй и не исправляй факты. Верни только распознанный текст."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _vision_data_url(path), "detail": "high"}},
                ],
            }
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    payload = {
        "rec_texts": [line.strip() for line in text.splitlines() if line.strip()],
        "engine": "vision_llm",
        "model": model,
        "latency_s": round(time.perf_counter() - started, 2),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"page": path.name, "latency_s": payload["latency_s"], "characters": len(text)}


def run_vision_ocr(
    manifest_path: Path,
    credentials_file: Path,
    model: str = "openai/gpt-4o-mini",
    workers: int = 4,
) -> Path:
    queue_path = prepare_ocr_queue(manifest_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    input_dir = queue_path.parent / "input"
    output_dir = queue_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for item in queue:
        source = input_dir / item["input_name"]
        expected = output_dir / f"{source.stem}_res.json"
        if not expected.exists():
            pending.append((source, expected))
    if not pending:
        set_stage(manifest_path, "fast_ocr", "completed", completed_pages=len(queue), elapsed_s=0, cache_hits=len(queue))
        return output_dir

    creds = read_credentials(credentials_file)
    client = OpenAI(api_key=creds["OPENAI_API_KEY"], base_url=creds.get("BASE_URL"), timeout=120)
    set_stage(manifest_path, "fast_ocr", "running", pending_pages=len(pending), workers=workers, engine="vision_llm")
    started = time.perf_counter()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_run_vision_page, source, destination, client, model): source.name
            for source, destination in pending
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"page": futures[future], "error": type(error).__name__, "detail": str(error)[:300]})
    elapsed = round(time.perf_counter() - started, 2)
    if errors:
        set_stage(manifest_path, "fast_ocr", "failed", elapsed_s=elapsed, failed_pages=errors)
        raise RuntimeError(f"Vision OCR failed for {len(errors)} pages")
    completed_count = len(list(output_dir.glob("*_res.json")))
    set_stage(
        manifest_path,
        "fast_ocr",
        "completed",
        elapsed_s=elapsed,
        completed_pages=completed_count,
        average_page_latency_s=round(sum(item["latency_s"] for item in results) / len(results), 2),
        engine="vision_llm",
        model=model,
    )
    return output_dir


def _run_api_page(
    path: Path,
    output_path: Path,
    api_url: str,
    timeout_s: int = 180,
) -> dict[str, Any]:
    started = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    with path.open("rb") as handle:
        response = session.post(
            f"{api_url.rstrip('/')}/v1/ocr",
            files={"file": (path.name, handle, "image/jpeg")},
            timeout=timeout_s,
        )
    response.raise_for_status()
    remote = response.json()
    lines = remote.get("lines") or []
    payload = {
        "rec_texts": [str(item.get("text") or "") for item in lines],
        "rec_scores": [item.get("score") for item in lines],
        "rec_polys": [item.get("polygon") for item in lines],
        "engine": "paddleocr_api",
        "model": remote.get("model"),
        "device": remote.get("device"),
        "remote_latency_s": remote.get("latency_s"),
        "latency_s": round(time.perf_counter() - started, 3),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "page": path.name,
        "latency_s": payload["latency_s"],
        "remote_latency_s": payload["remote_latency_s"],
        "characters": sum(len(value) for value in payload["rec_texts"]),
    }


def run_api_ocr(
    manifest_path: Path,
    api_url: str,
    workers: int = 4,
) -> Path:
    queue_path = prepare_ocr_queue(manifest_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    input_dir = queue_path.parent / "input"
    output_dir = queue_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for item in queue:
        source = input_dir / item["input_name"]
        expected = output_dir / f"{source.stem}_res.json"
        if not expected.exists():
            pending.append((source, expected))
    if not pending:
        set_stage(
            manifest_path,
            "fast_ocr",
            "completed",
            completed_pages=len(queue),
            elapsed_s=0,
            cache_hits=len(queue),
            engine="paddleocr_api",
        )
        return output_dir

    session = requests.Session()
    session.trust_env = False
    health = session.get(f"{api_url.rstrip('/')}/health", timeout=10)
    health.raise_for_status()
    set_stage(
        manifest_path,
        "fast_ocr",
        "running",
        pending_pages=len(pending),
        workers=workers,
        engine="paddleocr_api",
        api_url=api_url,
    )
    started = time.perf_counter()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_run_api_page, source, destination, api_url): source.name
            for source, destination in pending
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {"page": futures[future], "error": type(error).__name__, "detail": str(error)[:300]}
                )
    elapsed = round(time.perf_counter() - started, 2)
    if errors:
        set_stage(manifest_path, "fast_ocr", "failed", elapsed_s=elapsed, failed_pages=errors)
        raise RuntimeError(f"PaddleOCR API failed for {len(errors)} pages")
    set_stage(
        manifest_path,
        "fast_ocr",
        "completed",
        elapsed_s=elapsed,
        completed_pages=len(list(output_dir.glob("*_res.json"))),
        average_page_latency_s=round(sum(item["latency_s"] for item in results) / len(results), 3),
        engine="paddleocr_api",
        api_url=api_url,
    )
    return output_dir


def _split_ocr_lines(lines: list[str], max_chars: int = 6000) -> list[list[str]]:
    parts: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        if current and current_chars + len(line) + 1 > max_chars:
            parts.append(current)
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line) + 1
    if current:
        parts.append(current)
    return parts


def build_provenance_markdown(manifest_path: Path) -> Path:
    manifest = load_manifest(manifest_path)
    queue_path = manifest_path.parent / "fast_ocr" / "queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    output_dir = queue_path.parent / "output"
    parts = [f"# Report {manifest['report_id']} fast OCR", ""]
    included = 0
    for item in queue:
        result_path = output_dir / f"{Path(item['input_name']).stem}_res.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        texts = [str(value).strip() for value in payload.get("rec_texts", []) if str(value).strip()]
        marker = f"[SOURCE_PAGE: {item['page_id']}; path={item['source_path']}]"
        marked_texts = [f"{marker} {line}" for line in texts]
        for part_index, text_part in enumerate(_split_ocr_lines(marked_texts), 1):
            suffix = f" part:{part_index}" if len(marked_texts) > len(text_part) else ""
            parts.extend([f"## {item['page_id']}{suffix}", ""])
            parts.extend(text_part)
            parts.append("")
        included += 1
    if not included:
        raise RuntimeError("No OCR results are available for Markdown assembly")
    markdown = manifest_path.parent / f"report_{manifest['report_id']}_fast_ocr.md"
    markdown.write_text("\n".join(parts), encoding="utf-8")
    return markdown


def split_markdown_documents(markdown: Path, max_chars: int = 20_000) -> list[Path]:
    output_dir = markdown.parent / "ragflow_documents"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob(f"{markdown.stem}_part_*.md"):
        stale.unlink()
    lines = markdown.read_text(encoding="utf-8").splitlines()
    documents: list[Path] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        line_chars = len(line) + 1
        if current and current_chars + line_chars > max_chars:
            path = output_dir / f"{markdown.stem}_part_{len(documents) + 1:03d}.md"
            path.write_text("\n".join(current) + "\n", encoding="utf-8")
            documents.append(path)
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_chars
    if current:
        path = output_dir / f"{markdown.stem}_part_{len(documents) + 1:03d}.md"
        path.write_text("\n".join(current) + "\n", encoding="utf-8")
        documents.append(path)
    return documents


class RagflowIngestor:
    def __init__(self, base_url: str, token: str, proxy: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})

    def ingest(self, document: Path, dataset_id: str, timeout_s: int = 300) -> dict[str, Any]:
        digest = hashlib.sha256(document.read_bytes()).hexdigest()[:12]
        remote_name = f"{document.stem}_{digest}{document.suffix}"
        listing = self.session.get(
            f"{self.base_url}/datasets/{dataset_id}/documents",
            params={"page": 1, "page_size": 100, "keywords": remote_name},
            verify=False,
            timeout=30,
        )
        listing.raise_for_status()
        docs = (listing.json().get("data") or {}).get("docs") or []
        match = next((item for item in docs if item.get("name") == remote_name), None)
        if match:
            document_id = match["id"]
            uploaded = False
        else:
            with document.open("rb") as handle:
                response = self.session.post(
                    f"{self.base_url}/datasets/{dataset_id}/documents",
                    files={"file": (remote_name, handle, "text/markdown")},
                    verify=False,
                    timeout=60,
                )
            response.raise_for_status()
            document_id = response.json()["data"][0]["id"]
            uploaded = True
        parse = self.session.post(
            f"{self.base_url}/datasets/{dataset_id}/chunks",
            json={"document_ids": [document_id]},
            verify=False,
            timeout=60,
        )
        parse.raise_for_status()
        deadline = time.monotonic() + timeout_s
        state = {}
        while time.monotonic() < deadline:
            response = self.session.get(
                f"{self.base_url}/datasets/{dataset_id}/documents",
                params={"page": 1, "page_size": 100, "keywords": remote_name},
                verify=False,
                timeout=30,
            )
            response.raise_for_status()
            docs = (response.json().get("data") or {}).get("docs") or []
            state = next((item for item in docs if item.get("id") == document_id), {})
            if state.get("run") == "DONE":
                break
            if state.get("run") == "FAIL":
                raise RuntimeError(state.get("progress_msg") or "RAGFlow parsing failed")
            time.sleep(2)
        else:
            raise TimeoutError(f"RAGFlow did not parse {remote_name} within {timeout_s}s")
        return {"uploaded": uploaded, "remote_name": remote_name, "document_id": document_id, "state": state}

    def retrieve(
        self,
        dataset_id: str,
        document_id: str | list[str],
        question: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        document_ids = document_id if isinstance(document_id, list) else [document_id]
        response = self.session.post(
            f"{self.base_url}/retrieval",
            json={
                "dataset_ids": [dataset_id],
                "document_ids": document_ids,
                "question": question,
                "page": 1,
                "page_size": limit,
                "similarity_threshold": 0.05,
                "vector_similarity_weight": 0.3,
                "top_k": 64,
                "keyword": True,
            },
            verify=False,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in {0, None}:
            raise RuntimeError(payload.get("message") or f"RAGFlow code {payload.get('code')}")
        return (payload.get("data") or {}).get("chunks") or []


def ingest_ragflow(
    manifest_path: Path,
    markdown: Path,
    dataset_id: str,
    token_file: Path,
    base_url: str,
    proxy: str | None,
) -> Path:
    set_stage(manifest_path, "ragflow_ingest", "running")
    started = time.perf_counter()
    try:
        ingestor = RagflowIngestor(base_url, read_token(token_file), proxy)
        documents = split_markdown_documents(markdown)
        ingested = [ingestor.ingest(document, dataset_id) for document in documents]
        document_ids = [item["document_id"] for item in ingested]
        result = {
            "uploaded": any(item["uploaded"] for item in ingested),
            "documents": ingested,
            "document_ids": document_ids,
            "document_id": document_ids[0],
            "state": {
                "run": "DONE",
                "chunk_count": sum(int((item.get("state") or {}).get("chunk_count") or 0) for item in ingested),
            },
        }
    except Exception as error:
        set_stage(manifest_path, "ragflow_ingest", "failed", error=type(error).__name__)
        raise
    result["dataset_id"] = dataset_id
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    output = manifest_path.parent / "ragflow.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    set_stage(
        manifest_path,
        "ragflow_ingest",
        "completed",
        elapsed_s=result["elapsed_s"],
        document_id=result["document_id"],
        document_count=len(result["document_ids"]),
        chunk_count=result["state"].get("chunk_count"),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable report fast-path stages")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("stage", choices=["prepare", "ocr", "vision-ocr", "markdown", "ingest", "fast", "fast-vision"])
    parser.add_argument("--paddleocr", type=Path)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--vision-model", default="openai/gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset-id")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--base-url", default="https://ragflow-dev.finam.ru/api/v1")
    parser.add_argument("--proxy", default="socks5h://127.0.0.1:7777")
    args = parser.parse_args()

    if args.stage == "prepare":
        print(prepare_ocr_queue(args.manifest))
        return
    output_dir = args.manifest.parent / "fast_ocr" / "output"
    if args.stage in {"ocr", "fast"}:
        if not args.paddleocr:
            parser.error("--paddleocr is required for OCR")
        output_dir = run_fast_ocr(args.manifest, args.paddleocr, args.workers)
        if args.stage == "ocr":
            print(output_dir)
            return
    if args.stage in {"vision-ocr", "fast-vision"}:
        if not args.credentials_file:
            parser.error("--credentials-file is required for vision OCR")
        output_dir = run_vision_ocr(args.manifest, args.credentials_file, args.vision_model, args.workers)
        if args.stage == "vision-ocr":
            print(output_dir)
            return
    markdown = build_provenance_markdown(args.manifest)
    if args.stage == "markdown":
        print(markdown)
        return
    if args.stage in {"ingest", "fast", "fast-vision"}:
        if not args.dataset_id or not args.token_file:
            parser.error("--dataset-id and --token-file are required for RAGFlow ingest")
        print(ingest_ragflow(args.manifest, markdown, args.dataset_id, args.token_file, args.base_url, args.proxy))


if __name__ == "__main__":
    main()
