from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from rag_service import RagflowClient, ReportConfig, read_key_values, read_token


DOCUMENT_RE = re.compile(
    r"^report_(.+?)_fast_ocr(?:_part_\d+)?(?:_[0-9a-f]+)?\.md$",
    re.IGNORECASE,
)
SOURCE_RE = re.compile(r"\[источник\s+(\d+)\]", re.IGNORECASE)
SYSTEM_PROMPT = """Ты evidence-ассистент по фонду архивных геологических отчётов.
Отвечай только по предоставленным фрагментам. Для каждого факта указывай номер
отчёта и ссылку вида [источник 1]. Объединяй дубли из частей одного отчёта.
Если OCR неоднозначен или данных недостаточно, прямо скажи об этом. Ответь кратко
и по-русски."""


def report_id_from_document(name: str | None) -> str | None:
    match = DOCUMENT_RE.match(name or "")
    return match.group(1) if match else None


class CorpusSearchService:
    def __init__(
        self,
        configs: list[ReportConfig],
        retrievers: list[Any] | None = None,
        llm: OpenAI | None = None,
        workers: int = 24,
    ) -> None:
        if not configs:
            raise ValueError("No report configs for corpus search")
        self.workers = max(1, workers)
        primary = configs[0]
        if retrievers is None:
            proxy = os.environ.get("RAGFLOW_PROXY", "socks5h://127.0.0.1:7777")
            if proxy.lower() in {"", "none", "off"}:
                proxy = None
            retrievers = []
            for config in configs:
                metadata = json.loads(config.ragflow_metadata_path.read_text(encoding="utf-8"))
                state = metadata.get("document_state", {})
                dataset_id = metadata.get("dataset_id") or state.get("dataset_id")
                document_ids = list(metadata.get("document_ids") or [])
                if not document_ids and metadata.get("document_id"):
                    document_ids = [str(metadata["document_id"])]
                if not document_ids:
                    document_ids = [
                        str(item["document_id"])
                        for item in metadata.get("documents", [])
                        if item.get("document_id")
                    ]
                if dataset_id and document_ids:
                    retrievers.append(RagflowClient(
                        token=read_token(config.ragflow_token_path),
                        dataset_id=str(dataset_id),
                        document_ids=document_ids,
                        proxy_url=proxy,
                    ))
        if not retrievers:
            raise ValueError("RAGFlow corpus has no indexed documents")
        self.retrievers = retrievers
        credentials = read_key_values(primary.llm_credentials_path)
        self.model = os.environ.get("DEMO_LLM_MODEL", "openai/gpt-4o-mini")
        self.llm = llm or OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url=credentials.get("BASE_URL"),
            timeout=60,
        )

    def retrieve(self, question: str, limit: int = 12) -> dict[str, Any]:
        started = time.perf_counter()
        results = []
        errors = []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(self.retrievers))) as pool:
            futures = {
                pool.submit(retriever.retrieve, question, min(5, limit)): retriever
                for retriever in self.retrievers
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    errors.append(type(error).__name__)
        if not results and errors:
            raise RuntimeError(f"RAGFlow corpus retrieval failed: {sorted(set(errors))}")

        unique = {}
        for result in results:
            for chunk in result.get("chunks", []):
                key = (chunk.get("document_name"), chunk.get("raw_content"))
                current = unique.get(key)
                if current is None or chunk.get("similarity", 0) > current.get("similarity", 0):
                    unique[key] = chunk
        chunks = sorted(
            unique.values(),
            key=lambda item: (-float(item.get("similarity") or 0), str(item.get("document_name") or "")),
        )[:limit]
        for rank, chunk in enumerate(chunks, start=1):
            chunk["rank"] = rank
            chunk["report_id"] = report_id_from_document(chunk.get("document_name"))
        return {
            "chunks": chunks,
            "total": len(unique),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "transport": "ragflow_fanout",
            "partial_errors": errors,
        }

    def ask(self, question: str, allow_llm: bool = True) -> dict[str, Any]:
        question = question.strip()
        if len(question) < 5:
            raise ValueError("Question is too short")
        if len(question) > 1000:
            raise ValueError("Question is too long")
        retrieved = self.retrieve(question)
        chunks = retrieved["chunks"]
        evidence = [{key: value for key, value in chunk.items() if key != "raw_content"} for chunk in chunks]
        base = {
            "question": question,
            "evidence": evidence,
            "retrieval_latency_ms": retrieved["latency_ms"],
            "retrieval_transport": retrieved["transport"],
            "partial_errors": retrieved["partial_errors"],
        }
        if not chunks:
            return {**base, "answer": "RAGFlow не нашёл релевантных фрагментов по фонду.", "source": "ragflow", "model": None}
        if not allow_llm:
            return {**base, "answer": "Найдены релевантные фрагменты RAGFlow; генерация отключена.", "source": "ragflow_retrieval", "model": None}

        context = []
        for chunk in chunks:
            pages = ", ".join(page.split(":")[1] for page in chunk.get("evidence", [])) or "не определены"
            context.append(
                f"ИСТОЧНИК {chunk['rank']} (отчёт {chunk.get('report_id') or 'не определён'}, "
                f"страницы {pages}):\n{chunk['excerpt']}"
            )
        generation_started = time.perf_counter()
        generation_error = None
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"ВОПРОС:\n{question}\n\n" + "\n\n".join(context)},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as error:
            generation_error = type(error).__name__
            answer = "\n\n".join(
                f"Отчёт {chunk.get('report_id') or 'не определён'}: {chunk['excerpt'][:500]} "
                f"[источник {chunk['rank']}]"
                for chunk in chunks[:3]
            )
        cited = {int(value) for value in SOURCE_RE.findall(answer)}
        invalid = sorted(cited - set(range(1, len(chunks) + 1)))
        return {
            **base,
            "answer": answer,
            "source": "ragflow_llm" if generation_error is None else "ragflow_extractive_fallback",
            "model": self.model if generation_error is None else None,
            "generation_error": generation_error,
            "generation_latency_ms": round((time.perf_counter() - generation_started) * 1000),
            "citation_qc": {
                "status": "pass" if cited and not invalid else "review",
                "cited_sources": sorted(cited),
                "invalid": invalid,
                "missing_citations": not cited,
            },
        }
