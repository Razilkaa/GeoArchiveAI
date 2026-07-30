from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from app.config import settings


SOURCE_PAGE_RE = re.compile(
    r"\[SOURCE_PAGE:\s*(?:page_(?P<legacy>\d{5})\.[^\]]+|page:(?P<current>\d{5})(?:;[^\]]*)?)\]"
)
CITATION_RE = re.compile(r"\[стр\.\s*([\d\s,;–—-]+)\]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё-]{2,}")
STOPWORDS = {
    "как", "для", "или", "при", "что", "это", "были", "была", "было", "быть",
    "какие", "какой", "какая", "где", "когда", "после", "перед", "между", "через",
    "отчёт", "отчет", "отчёте", "отчете", "работ", "работы", "страница", "страницы",
}
RUSSIAN_SUFFIXES = tuple(
    sorted(
        {
            "иями", "ями", "ами", "его", "ого", "ему", "ому", "ыми", "ими",
            "ую", "юю", "ая", "яя", "ое", "ее", "ие", "ые", "ой", "ей",
            "ам", "ям", "ах", "ях", "ов", "ев", "ом", "ем", "ы", "и", "а",
            "я", "у", "ю", "е",
        },
        key=len,
        reverse=True,
    )
)

SYSTEM_PROMPT = """Ты evidence-ассистент по архивному геологическому отчёту.
Отвечай только по предоставленным фрагментам. Не дополняй ответ общими знаниями.
После каждого фактического утверждения ставь ссылку вида [стр. 14].
Если данных недостаточно, прямо скажи об этом. Учитывай OCR-ошибки и не исправляй
неоднозначные имена без пояснения. Проверенные структурированные факты имеют
приоритет над сырым OCR-текстом, особенно для идентификаторов скважин и названий.
Ответ должен быть кратким и на русском языке."""


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def read_token(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.split("=", 1)[-1].strip().strip('"').strip("'")
    raise ValueError(f"No token found in {path}")


def clean_content(content: str, max_chars: int = 2200) -> str:
    lines: list[str] = []
    for raw_line in content.replace("\r", "").splitlines():
        line = SOURCE_PAGE_RE.sub("", raw_line).strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def page_ids(content: str) -> list[str]:
    pages = {
        int(match.group("legacy") or match.group("current"))
        for match in SOURCE_PAGE_RE.finditer(content)
    }
    return [f"text:{page:05d}" for page in sorted(pages)]


def cited_page_numbers(answer: str) -> set[int]:
    pages: set[int] = set()
    for citation in CITATION_RE.findall(answer):
        pages.update(int(value) for value in re.findall(r"\d{1,5}", citation))
    return pages


def retrieval_source(transport: str | None, suffix: str = "") -> str:
    base = "ragflow"
    return f"{base}_{suffix}" if suffix else base


def tokens(text: str) -> list[str]:
    result = []
    for raw_token in TOKEN_RE.findall(text):
        token = raw_token.lower().replace("ё", "е")
        if token in STOPWORDS:
            continue
        if re.fullmatch(r"[а-я-]+", token):
            for suffix in RUSSIAN_SUFFIXES:
                if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                    token = token[: -len(suffix)]
                    break
        result.append(token)
    return result


@dataclass(frozen=True)
class ReportConfig:
    report_id: str
    bundle_path: Path
    ragflow_metadata_path: Path
    local_corpus_path: Path
    ragflow_token_path: Path
    llm_credentials_path: Path

    @classmethod
    def from_dict(cls, report_id: str, value: dict[str, str]) -> "ReportConfig":
        return cls(report_id=report_id, **{key: Path(path) for key, path in value.items()})


class RagflowClient:
    def __init__(
        self,
        token: str,
        dataset_id: str,
        document_ids: list[str] | None = None,
        base_url: str | None = None,
        proxy_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.token = token
        self.dataset_id = dataset_id
        self.document_ids = document_ids or []
        self.base_url = (base_url or settings.ragflow_url).rstrip("/")
        self.proxy_url = proxy_url
        self.session = session or requests.Session()
        if session is None:
            self.session.trust_env = False

    def retrieve(self, question: str, limit: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        limit = limit or settings.ragflow_page_size
        proxies = None
        if self.proxy_url:
            proxies = {"http": self.proxy_url, "https": self.proxy_url}
        request_kwargs = {
            "headers": {"Authorization": f"Bearer {self.token}"},
            "json": {
                "dataset_ids": [self.dataset_id],
                "document_ids": self.document_ids,
                "question": question,
                "page": 1,
                "page_size": limit,
                "similarity_threshold": settings.ragflow_similarity_threshold,
                "vector_similarity_weight": settings.ragflow_vector_similarity_weight,
                "top_k": settings.ragflow_top_k,
                # RAGFlow 0.20 treats the boolean keyword flag as an expensive
                # keyword-generation request. On the local deployment it
                # stalls for minutes; hybrid retrieval itself does not require
                # this optional flag.
                "keyword": False,
            },
            "verify": settings.ragflow_verify_tls,
            "timeout": settings.ragflow_retrieval_timeout_s,
        }
        transport = "proxy" if proxies else "direct"
        active_proxies = proxies
        response: requests.Response | None = None
        for attempt in range(settings.ragflow_retrieval_attempts):
            try:
                response = self.session.post(
                    f"{self.base_url}/retrieval",
                    proxies=active_proxies,
                    **request_kwargs,
                )
                response.raise_for_status()
                break
            except requests.ConnectionError:
                if active_proxies:
                    active_proxies = None
                    transport = "direct_fallback"
                elif attempt + 1 >= settings.ragflow_retrieval_attempts:
                    raise
            except requests.Timeout:
                if attempt + 1 >= settings.ragflow_retrieval_attempts:
                    raise
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status not in {429, 502, 503, 504} or attempt + 1 >= settings.ragflow_retrieval_attempts:
                    raise
            time.sleep(settings.ragflow_retry_backoff_s * (attempt + 1))
        if response is None:
            raise RuntimeError("RAGFlow retrieval did not return a response")
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"RAGFlow error: {payload.get('message') or payload.get('code')}")

        chunks = []
        for rank, chunk in enumerate((payload.get("data") or {}).get("chunks") or [], start=1):
            content = chunk.get("content_with_weight") or chunk.get("content") or ""
            chunks.append(
                {
                    "rank": rank,
                    "similarity": round(float(chunk.get("similarity") or 0.0), 4),
                    "document_name": chunk.get("document_name") or chunk.get("document_keyword"),
                    "evidence": page_ids(content),
                    "excerpt": clean_content(content),
                    "raw_content": content,
                }
            )
        return {
            "chunks": chunks,
            "total": (payload.get("data") or {}).get("total"),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "transport": transport,
        }


def extractive_answer(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "В RAGFlow не найдено достаточно данных для ответа."
    parts = ["Внешняя модель недоступна. Наиболее релевантные фрагменты отчёта:"]
    for chunk in chunks[:2]:
        page = int(chunk["evidence"][0].split(":")[1])
        excerpt = chunk["excerpt"].replace("\n", " ")[:500].rstrip()
        parts.append(f"{excerpt} [стр. {page}]")
    return "\n\n".join(parts)


class ReportAnswerService:
    def __init__(
        self,
        config: ReportConfig,
        ragflow: RagflowClient | None = None,
        llm: OpenAI | None = None,
    ) -> None:
        self.config = config
        self.bundle = json.loads(config.bundle_path.read_text(encoding="utf-8"))
        metadata = json.loads(config.ragflow_metadata_path.read_text(encoding="utf-8"))
        document_state = metadata.get("document_state", {})
        dataset_id = metadata.get("dataset_id") or document_state.get("dataset_id")
        if not dataset_id:
            raise ValueError(f"RAGFlow dataset is missing in {config.ragflow_metadata_path}")
        document_ids = list(metadata.get("document_ids") or [])
        if not document_ids and metadata.get("document_id"):
            document_ids = [str(metadata["document_id"])]
        proxy = settings.proxy_url
        self.ragflow = ragflow or RagflowClient(
            token=read_token(config.ragflow_token_path),
            dataset_id=dataset_id,
            document_ids=document_ids,
            base_url=settings.ragflow_url,
            proxy_url=proxy,
        )
        credentials = read_key_values(config.llm_credentials_path)
        self.model = settings.llm_model
        self.llm = llm or OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url=credentials.get("BASE_URL"),
            timeout=60,
        )
        self.answer_cache: dict[str, dict[str, Any]] = {}

    def saved_answer(self, question: str) -> dict[str, Any] | None:
        normalized = question.strip().casefold()
        for item in self.bundle.get("preset_queries", []):
            if item["question"].strip().casefold() == normalized:
                return {**item, "source": "saved", "citation_qc": {"status": "pass", "invalid": []}}
        return None

    def structured_context(self, question: str) -> list[dict[str, Any]]:
        query_terms = set(tokens(question))
        context: list[dict[str, Any]] = []
        entities = self.bundle.get("entities", {})
        if any(term.startswith("скваж") for term in query_terms):
            context.extend(entities.get("wells", []))
        if any(term.startswith("структур") for term in query_terms):
            context.extend(entities.get("structures", []))
        if any(term.startswith("горизонт") for term in query_terms):
            context.extend(entities.get("horizons", []))
        return context

    def ask(self, question: str, allow_llm: bool = True) -> dict[str, Any]:
        question = question.strip()
        if len(question) < 5:
            raise ValueError("Question is too short")
        if len(question) > 1000:
            raise ValueError("Question is too long")
        cache_key = question.casefold()
        if allow_llm and cache_key in self.answer_cache:
            cached = copy.deepcopy(self.answer_cache[cache_key])
            cached["cache_hit"] = True
            return cached

        retrieved = self.ragflow.retrieve(question)
        transport = retrieved.get("transport")
        chunks = retrieved["chunks"]
        if not chunks:
            return {
                "question": question,
                "answer": "В индексированном отчёте не найдено достаточно данных для ответа.",
                "evidence": [],
                "source": retrieval_source(transport),
                "model": None,
                "retrieval_latency_ms": retrieved["latency_ms"],
                "retrieval_transport": retrieved.get("transport"),
                "generation_latency_ms": 0,
                "citation_qc": {"status": "pass", "invalid": []},
            }

        if not allow_llm:
            return {
                "question": question,
                "answer": "Найдены релевантные фрагменты; генерация ответа отключена.",
                "evidence": [{key: value for key, value in chunk.items() if key != "raw_content"} for chunk in chunks],
                "source": retrieval_source(transport, "retrieval"),
                "model": None,
                "retrieval_latency_ms": retrieved["latency_ms"],
                "retrieval_transport": retrieved.get("transport"),
                "generation_latency_ms": 0,
                "citation_qc": {"status": "pass", "invalid": []},
            }

        context_parts = []
        for chunk in chunks:
            pages = ", ".join(str(int(page.split(":")[1])) for page in chunk["evidence"])
            context_parts.append(f"ФРАГМЕНТ {chunk['rank']} (страницы: {pages}):\n{chunk['excerpt']}")
        structured = self.structured_context(question)
        structured_text = json.dumps(structured, ensure_ascii=False, indent=2)
        valid_pages = {int(page.split(":")[1]) for chunk in chunks for page in chunk["evidence"]}
        valid_pages.update(
            int(page.split(":")[1]) for item in structured for page in item.get("evidence", [])
        )
        started = time.perf_counter()
        generation_error = None
        generation_attempts = 0
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ВОПРОС:\n{question}\n\n"
                    f"РАЗРЕШЁННЫЕ СТРАНИЦЫ ДЛЯ ССЫЛОК: {sorted(valid_pages)}\n\n"
                    f"ПРОВЕРЕННЫЕ СТРУКТУРИРОВАННЫЕ ФАКТЫ:\n{structured_text}\n\n"
                    "OCR/RAG-ФРАГМЕНТЫ:\n" + "\n\n".join(context_parts)
                ),
            },
        ]
        try:
            generation_attempts = 1
            response = self.llm.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
            )
            answer = (response.choices[0].message.content or "").strip()
            initial_citations = cited_page_numbers(answer)
            initial_invalid = sorted(initial_citations - valid_pages)
            if initial_invalid or not initial_citations:
                generation_attempts = 2
                repair_messages = messages + [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "Citation-QC отклонил ответ. Перепиши его без добавления новых "
                            f"фактов. Используй ссылки только на страницы {sorted(valid_pages)}; "
                            "для идентификаторов скважин используй structured facts. "
                            "Не добавляй итоговое утверждение без ссылки."
                        ),
                    },
                ]
                response = self.llm.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=repair_messages,
                )
                answer = (response.choices[0].message.content or "").strip()
        except Exception as error:
            generation_error = type(error).__name__
            answer = extractive_answer(chunks)
        generation_latency = round((time.perf_counter() - started) * 1000)

        cited_pages = cited_page_numbers(answer)
        invalid_pages = sorted(cited_pages - valid_pages)
        missing_citations = bool(answer) and not cited_pages
        qc_status = "pass" if not invalid_pages and not missing_citations else "review"

        result = {
            "question": question,
            "answer": answer,
            "evidence": [{key: value for key, value in chunk.items() if key != "raw_content"} for chunk in chunks],
            "source": retrieval_source(retrieved.get("transport"), "llm") if generation_error is None else "extractive_fallback",
            "model": self.model if generation_error is None else None,
            "generation_error": generation_error,
            "generation_attempts": generation_attempts,
            "structured_context": structured,
            "retrieval_latency_ms": retrieved["latency_ms"],
            "retrieval_transport": retrieved.get("transport"),
            "generation_latency_ms": generation_latency,
            "citation_qc": {
                "status": qc_status,
                "cited_pages": sorted(cited_pages),
                "invalid": invalid_pages,
                "missing_citations": missing_citations,
            },
            "cache_hit": False,
        }
        self.answer_cache[cache_key] = copy.deepcopy(result)
        return result


def load_report_configs(path: Path | None = None) -> dict[str, ReportConfig]:
    config_path = path or Path(__file__).with_name("reports.json")
    if not config_path.exists():
        return {}
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return {report_id: ReportConfig.from_dict(report_id, value) for report_id, value in raw.items()}
