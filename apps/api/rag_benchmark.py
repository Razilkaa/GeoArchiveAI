from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from rag_service import RagflowClient, read_token


SOURCE_RE = re.compile(r"\[SOURCE_PAGE:\s*page:(\d{5})(?:;[^\]]*)?\]\s*(.*)")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold().replace("ё", "е")))


def markdown_pages(markdown: Path) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    for line in markdown.read_text(encoding="utf-8").splitlines():
        match = SOURCE_RE.search(line)
        if match:
            pages.setdefault(int(match.group(1)), []).append(match.group(2))
    return {page: "\n".join(lines) for page, lines in pages.items()}


def group_matches(text: str, groups: list[list[str]]) -> list[bool]:
    normalized = normalize(text)
    return [any(normalize(variant) in normalized for variant in group) for group in groups]


def ratio(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 1.0


def evaluate_case(
    case: dict[str, Any],
    pages: dict[int, str],
    retrieved: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_pages = [int(page) for page in case["expected_pages"]]
    present_pages = [page for page in expected_pages if page in pages]
    expected_text = "\n".join(pages.get(page, "") for page in expected_pages)
    ocr_matches = group_matches(expected_text, case.get("term_groups", []))

    chunks = (retrieved or {}).get("chunks", [])
    relevant_rank = None
    for chunk in chunks:
        evidence_pages = {int(item.split(":", 1)[1]) for item in chunk.get("evidence", [])}
        if evidence_pages.intersection(expected_pages):
            relevant_rank = int(chunk["rank"])
            break
    retrieved_text = "\n".join(chunk.get("excerpt", "") for chunk in chunks)
    retrieved_matches = group_matches(retrieved_text, case.get("term_groups", []))

    retrieval_error = (retrieved or {}).get("error")
    if not present_pages:
        diagnosis = "ocr_page_missing"
    elif not all(ocr_matches):
        diagnosis = "ocr_content_loss"
    elif retrieval_error:
        diagnosis = "retrieval_error"
    elif retrieved is not None and relevant_rank is None:
        diagnosis = "retrieval_miss"
    elif retrieved is not None and not all(retrieved_matches):
        diagnosis = "retrieval_partial_context"
    else:
        diagnosis = "pass"

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_pages": expected_pages,
        "ocr": {
            "present_pages": present_pages,
            "page_recall": round(len(present_pages) / len(expected_pages), 4),
            "term_matches": ocr_matches,
            "term_recall": ratio(ocr_matches),
        },
        "retrieval": None
        if retrieved is None
        else {
            "first_relevant_rank": relevant_rank,
            "hit_at_5": relevant_rank is not None and relevant_rank <= 5,
            "reciprocal_rank": round(1 / relevant_rank, 4) if relevant_rank else 0.0,
            "term_matches": retrieved_matches,
            "term_recall": ratio(retrieved_matches),
            "latency_ms": retrieved.get("latency_ms"),
            "returned_chunks": len(chunks),
            "error": retrieval_error,
        },
        "diagnosis": diagnosis,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        item
        for item in results
        if item["ocr"]["page_recall"] == 1.0 and item["ocr"]["term_recall"] == 1.0
    ]
    retrieval = [item["retrieval"] for item in eligible if item["retrieval"] is not None]
    responded = [item for item in retrieval if not item.get("error")]
    counts: dict[str, int] = {}
    for item in results:
        counts[item["diagnosis"]] = counts.get(item["diagnosis"], 0) + 1
    return {
        "cases": len(results),
        "ocr_page_recall": round(sum(item["ocr"]["page_recall"] for item in results) / len(results), 4),
        "ocr_term_recall": round(sum(item["ocr"]["term_recall"] for item in results) / len(results), 4),
        "retrieval_eligible_cases": len(eligible),
        "retrieval_hit_at_5": round(sum(item["hit_at_5"] for item in retrieval) / len(retrieval), 4)
        if retrieval
        else None,
        "retrieval_mrr": round(sum(item["reciprocal_rank"] for item in retrieval) / len(retrieval), 4)
        if retrieval
        else None,
        "retrieval_term_recall": round(sum(item["term_recall"] for item in retrieval) / len(retrieval), 4)
        if retrieval
        else None,
        "retrieval_error_rate": round(sum(bool(item.get("error")) for item in retrieval) / len(retrieval), 4)
        if retrieval
        else None,
        "retrieval_hit_at_5_when_responded": round(
            sum(item["hit_at_5"] for item in responded) / len(responded), 4
        )
        if responded
        else None,
        "diagnoses": counts,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# RAG benchmark: {payload['report_id']}",
        "",
        f"- Corpus page coverage: **{summary['corpus_page_coverage']:.1%}** "
        f"({summary['corpus_indexed_pages']}/{summary['corpus_expected_pages']})"
        if summary.get("corpus_page_coverage") is not None
        else "",
        f"- OCR page recall: **{summary['ocr_page_recall']:.1%}**",
        f"- OCR fact recall: **{summary['ocr_term_recall']:.1%}**",
        f"- Retrieval eligible cases: **{summary['retrieval_eligible_cases']}**",
        f"- Retrieval Hit@5: **{summary['retrieval_hit_at_5']:.1%}**" if summary["retrieval_hit_at_5"] is not None else "- Retrieval: skipped",
        f"- Retrieval error rate: **{summary['retrieval_error_rate']:.1%}**" if summary["retrieval_error_rate"] is not None else "",
        f"- Retrieval MRR: **{summary['retrieval_mrr']:.3f}**" if summary["retrieval_mrr"] is not None else "",
        "",
        "| Case | OCR pages | OCR facts | Rank | Diagnosis |",
        "|---|---:|---:|---:|---|",
    ]
    for item in payload["results"]:
        retrieval = item["retrieval"] or {}
        lines.append(
            f"| {item['id']} | {item['ocr']['page_recall']:.0%} | "
            f"{item['ocr']['term_recall']:.0%} | {retrieval.get('first_relevant_rank') or '-'} | "
            f"{item['diagnosis']} |"
        )
    return "\n".join(line for line in lines if line != "") + "\n"


def run_benchmark(
    benchmark: Path,
    markdown: Path,
    metadata_path: Path,
    token_path: Path,
    *,
    manifest_path: Path | None = None,
    skip_retrieval: bool = False,
    limit: int = 10,
    proxy: str | None = "socks5h://127.0.0.1:7777",
) -> dict[str, Any]:
    definition = json.loads(benchmark.read_text(encoding="utf-8"))
    pages = markdown_pages(markdown)
    expected_corpus_pages: set[int] = set()
    if manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_corpus_pages = {
            int(page["id"].split(":", 1)[1])
            for page in manifest["pages"]
            if page.get("role") in {"text", "toc"}
        }
    client = None
    if not skip_retrieval:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset_id = metadata.get("dataset_id") or (metadata.get("document_state") or {}).get("dataset_id")
        document_ids = metadata.get("document_ids") or ([metadata["document_id"]] if metadata.get("document_id") else [])
        client = RagflowClient(read_token(token_path), dataset_id, document_ids, proxy_url=proxy)

    started = time.perf_counter()
    results = []
    for case in definition["cases"]:
        retrieved = None
        if client:
            try:
                retrieved = client.retrieve(case["question"], limit=limit)
            except Exception as error:
                retrieved = {
                    "chunks": [],
                    "latency_ms": None,
                    "error": f"{type(error).__name__}: {error}",
                }
        results.append(evaluate_case(case, pages, retrieved))
    summary = aggregate(results)
    summary["corpus_expected_pages"] = len(expected_corpus_pages) or None
    summary["corpus_indexed_pages"] = len(expected_corpus_pages.intersection(pages)) if expected_corpus_pages else len(pages)
    summary["corpus_page_coverage"] = (
        round(len(expected_corpus_pages.intersection(pages)) / len(expected_corpus_pages), 4)
        if expected_corpus_pages
        else None
    )
    return {
        "schema_version": 1,
        "report_id": definition["report_id"],
        "benchmark": str(benchmark),
        "markdown_pages": len(pages),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "summary": summary,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure OCR coverage and RAGFlow retrieval quality")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("token", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--proxy", default="socks5h://127.0.0.1:7777")
    args = parser.parse_args()
    proxy = None if args.proxy.casefold() in {"", "none", "off"} else args.proxy
    payload = run_benchmark(
        args.benchmark,
        args.markdown,
        args.metadata,
        args.token,
        manifest_path=args.manifest,
        skip_retrieval=args.skip_retrieval,
        limit=args.limit,
        proxy=proxy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
