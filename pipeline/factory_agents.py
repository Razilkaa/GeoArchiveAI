from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from factory_runner import RagflowIngestor, load_manifest, read_token, set_stage


AGENTS = {
    "structures": {
        "queries": [
            "Какие структуры, поднятия и локальные объекты выявлены, оконтурены или подготовлены?",
            "По каким отражающим горизонтам построены структуры и какие структуры рекомендованы под глубокое бурение?",
        ],
        "fields": ["name", "status", "horizons", "description", "evidence"],
    },
    "wells": {
        "queries": [
            "Какие номера глубоких, структурных и параметрических скважин упомянуты в отчете?",
            "В каких скважинах были испытания, нефтегазопроявления, притоки нефти, газа или воды?",
            "Каковы глубины скважин, интервалы испытания и результаты опробования пластов?",
        ],
        "fields": ["id", "depth_m", "tests", "shows", "inflows", "description", "evidence"],
    },
    "maps": {
        "queries": [
            "Какие структурные и геологические карты входят в графические приложения отчета?",
            "Какие схемы профилей, сейсмические разрезы и карты изохрон перечислены в оглавлении приложений?",
            "Каковы номера приложений, отражающие горизонты и масштабы карт?",
        ],
        "fields": ["title", "kind", "sheet", "horizons", "scale", "description", "evidence"],
    },
    "key_results": {
        "queries": [
            "Каковы цель, объем, методика и качество выполненных геофизических работ?",
            "Каковы основные геологические результаты, выводы и практические рекомендации отчета?",
        ],
        "fields": ["topic", "statement", "evidence"],
    },
}

SYSTEM = """Ты извлекаешь проверяемые факты из OCR архивного геолого-геофизического отчета.
Используй только предоставленные фрагменты. Не исправляй неоднозначные названия молча.
Верни только JSON-объект вида {"items": [...], "notes": [...]}.
У каждого item обязательно evidence: массив точных идентификаторов из маркеров SOURCE_PAGE.
Если данных нет, верни пустой items. Не добавляй общие знания."""

AGENTS = {
    "structures": {
        "queries": [
            "Какие структуры, поднятия и локальные объекты выявлены, по каким горизонтам построены, как оконтурены и какие рекомендованы под глубокое бурение?"
        ],
        "fields": ["name", "status", "horizons", "description", "evidence"],
    },
    "wells": {
        "queries": [
            "Какие скважины с номерами упомянуты: тип, глубина, интервалы испытаний и опробования?",
            "В каких пронумерованных скважинах отмечены нефтегазопроявления, притоки нефти, газа или воды и каковы результаты испытаний?",
        ],
        "fields": ["id", "depth_m", "tests", "shows", "inflows", "description", "evidence"],
    },
    "maps": {
        "queries": [
            "Какие карты, схемы профилей и сейсмические разрезы входят в приложения: название, номер листа, горизонт и масштаб?"
        ],
        "fields": ["title", "kind", "sheet", "horizons", "scale", "description", "evidence"],
    },
    "key_results": {
        "queries": [
            "Каковы цель, объем и качество работ, основные геологические результаты, выводы и практические рекомендации отчета?"
        ],
        "fields": ["topic", "statement", "evidence"],
    },
}

SOURCE_RE = re.compile(r"\[SOURCE_PAGE:\s*([^;\]\s]+)", re.IGNORECASE)
LOCAL_KEYWORDS = {
    "wells": ("скв.", "скважин", "бурен", "испытан", "опробован", "приток"),
}


def read_credentials(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(fenced)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", fenced, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Agent response is not a JSON object")
    value.setdefault("items", [])
    value.setdefault("notes", [])
    return value


def chunk_content(chunk: dict[str, Any]) -> str:
    return chunk.get("content_with_weight") or chunk.get("content") or ""


def source_refs(chunks: list[dict[str, Any]]) -> set[str]:
    return {ref for chunk in chunks for ref in SOURCE_RE.findall(chunk_content(chunk))}


def local_keyword_chunks(markdown: Path, keywords: tuple[str, ...], limit: int = 8) -> list[dict[str, Any]]:
    if not markdown.exists():
        return []
    text = markdown.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^## page:)", text, flags=re.MULTILINE)
    scored = []
    for block in blocks:
        lowered = block.casefold()
        score = sum(lowered.count(keyword.casefold()) for keyword in keywords)
        refs = SOURCE_RE.findall(block)
        if score and refs:
            scored.append((score, refs[0], block.strip()))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"id": f"local:{ref}", "content": block, "source": "local_keyword"}
        for _, ref, block in scored[:limit]
    ]


def validate_evidence(result: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    invalid = set()
    missing = 0
    for item in result.get("items", []):
        evidence = item.get("evidence") or []
        if not evidence:
            missing += 1
        invalid.update(str(ref) for ref in evidence if str(ref) not in allowed)
    return {
        "status": "pass" if not invalid and not missing else "review",
        "invalid": sorted(invalid),
        "items_without_evidence": missing,
        "allowed_source_count": len(allowed),
    }


def domain_qc(name: str, result: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    qc = validate_evidence(result, allowed)
    if name == "wells":
        suspicious = []
        for item in result.get("items", []):
            well_id = str(item.get("id") or "").strip()
            if not re.search(r"\d", well_id):
                suspicious.append(well_id or "<missing>")
            elif re.search(r"[оОoO]", well_id):
                suspicious.append(well_id)
        qc["suspicious_ids"] = sorted(set(suspicious))
        if suspicious:
            qc["status"] = "review"
    return qc


def run_agent(
    name: str,
    spec: dict[str, Any],
    chunks: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    unique_chunks = {}
    for chunk in chunks:
        key = chunk.get("id") or chunk.get("chunk_id") or chunk_content(chunk)
        unique_chunks[key] = chunk
    chunks = list(unique_chunks.values())
    context = []
    for rank, chunk in enumerate(chunks, 1):
        context.append(f"FRAGMENT {rank}:\n{chunk_content(chunk)}")
    allowed = source_refs(chunks)
    prompt = (
        f"ЗАДАЧА: {' '.join(spec['queries'])}\n"
        f"ПОЛЯ КАЖДОГО ITEM: {spec['fields']}\n\n" + "\n\n".join(context)
    )
    if name == "wells":
        prompt += (
            "\n\nВАЖНО: id скважины обязан содержать хотя бы одну арабскую цифру. "
            "Название площади, структуры или населенного пункта без номера не является id скважины."
        )
    cache_key = hashlib.sha256(f"{model}\n{SYSTEM}\n{prompt}".encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"agent_{name}_{cache_key}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        cached["latency_s"] = round(time.perf_counter() - started, 2)
        return cached
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    )
    result = parse_json_object(response.choices[0].message.content or "{}")
    payload = {
        "status": "completed",
        "model": model,
        "latency_s": round(time.perf_counter() - started, 2),
        "cache_hit": False,
        "retrieved_chunks": len(chunks),
        "result": result,
        "evidence_qc": domain_qc(name, result, allowed),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def retrieve_with_retry(
    ragflow: RagflowIngestor,
    dataset_id: str,
    document_id: str | list[str],
    query: str,
    attempts: int = 3,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    document_key = "\n".join(document_id) if isinstance(document_id, list) else document_id
    cache_key = hashlib.sha256(f"{dataset_id}\n{document_key}\n{query}".encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"retrieval_{cache_key}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    for attempt in range(attempts):
        try:
            chunks = ragflow.retrieve(dataset_id, document_id, query, 8)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            return chunks
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    return []


def run_agents(
    manifest_path: Path,
    ragflow_metadata: Path,
    token_file: Path,
    credentials_file: Path,
    model: str,
    base_url: str,
    proxy: str | None,
) -> Path:
    manifest = load_manifest(manifest_path)
    metadata = json.loads(ragflow_metadata.read_text(encoding="utf-8"))
    dataset_id = (
        metadata.get("dataset_id")
        or (metadata.get("state") or {}).get("dataset_id")
        or (metadata.get("document_state") or {}).get("dataset_id")
    )
    document_id = metadata.get("document_ids") or metadata["document_id"]
    if not dataset_id:
        raise ValueError("RAGFlow metadata has no dataset_id")
    creds = read_credentials(credentials_file)
    client = OpenAI(api_key=creds["OPENAI_API_KEY"], base_url=creds.get("BASE_URL"), timeout=120)
    ragflow = RagflowIngestor(base_url, read_token(token_file), proxy)
    set_stage(manifest_path, "entity_agents", "running", agent_count=len(AGENTS))
    started = time.perf_counter()
    cache_dir = manifest_path.parent / "cache"
    results: dict[str, Any] = {}
    retrieved_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_errors: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                retrieve_with_retry,
                ragflow,
                dataset_id,
                document_id,
                query,
                3,
                cache_dir,
            ): (name, query)
            for name, spec in AGENTS.items()
            for query in spec["queries"]
        }
        for future in as_completed(futures):
            name, query = futures[future]
            try:
                retrieved_by_agent[name].extend(future.result())
            except Exception as error:
                retrieval_errors[name].append(f"{type(error).__name__}: {query}")

    local_markdown = manifest_path.parent / f"report_{manifest['report_id']}_fast_ocr.md"
    if ragflow_metadata.resolve().parent == manifest_path.resolve().parent:
        for name, keywords in LOCAL_KEYWORDS.items():
            retrieved_by_agent[name].extend(local_keyword_chunks(local_markdown, keywords))

    with ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
        futures = {
            pool.submit(run_agent, name, spec, retrieved_by_agent[name], client, model, cache_dir): name
            for name, spec in AGENTS.items()
            if retrieved_by_agent[name]
        }
        for name in AGENTS:
            if not retrieved_by_agent[name]:
                results[name] = {
                    "status": "failed",
                    "error": "RetrievalError",
                    "detail": "; ".join(retrieval_errors[name]) or "No chunks returned",
                }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                results[name]["retrieval_errors"] = retrieval_errors[name]
            except Exception as error:
                results[name] = {"status": "failed", "error": type(error).__name__, "detail": str(error)[:500]}
    elapsed = round(time.perf_counter() - started, 2)
    output = manifest_path.parent / "agent_results.json"
    payload = {
        "report_id": manifest["report_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "document_id": metadata["document_id"],
        "document_ids": metadata.get("document_ids") or [metadata["document_id"]],
        "elapsed_s": elapsed,
        "agents": results,
    }
    verification_queue = []
    for name, result in results.items():
        qc = result.get("evidence_qc", {})
        if qc.get("status") == "review":
            verification_queue.append(
                {
                    "agent": name,
                    "reason": "entity_qc_review",
                    "suspicious_ids": qc.get("suspicious_ids", []),
                    "source_pages": sorted(
                        {
                            str(ref)
                            for item in result.get("result", {}).get("items", [])
                            for ref in (item.get("evidence") or [])
                        }
                    ),
                }
            )
    payload["verification_queue"] = verification_queue
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [name for name, result in results.items() if result["status"] == "failed"]
    set_stage(
        manifest_path,
        "entity_agents",
        "failed" if failed else "completed",
        elapsed_s=elapsed,
        failed_agents=failed,
        item_counts={name: len(result.get("result", {}).get("items", [])) for name, result in results.items()},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel report extraction agents")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ragflow_metadata", type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--credentials-file", required=True, type=Path)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--base-url", default=os.environ.get("RAGFLOW_URL"))
    parser.add_argument("--proxy", default=os.environ.get("RAGFLOW_PROXY"))
    args = parser.parse_args()
    print(
        run_agents(
            args.manifest,
            args.ragflow_metadata,
            args.token_file,
            args.credentials_file,
            args.model,
            args.base_url,
            args.proxy,
        )
    )


if __name__ == "__main__":
    main()
