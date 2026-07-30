from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PAGE_RE = re.compile(r"(\d{1,5})$")
ABSTAIN_MARKERS = ("нет информации", "недостаточно данных", "не найдено", "не указано")


def page_number(value: str) -> int | None:
    match = PAGE_RE.search(value)
    return int(match.group(1)) if match else None


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


@dataclass
class CaseResult:
    id: str
    report_id: str
    status: str
    retrieval_hit: bool
    reciprocal_rank: float
    required_term_recall: float | None
    forbidden_term_pass: bool | None
    abstention_pass: bool | None
    citation_qc: str | None
    latency_ms: int
    evidence_pages: list[int]
    answer: str | None
    error: str | None


def evaluate_case(base_url: str, item: dict[str, Any], mode: str, timeout: float) -> CaseResult:
    started = time.perf_counter()
    expected = {page_number(str(value)) for value in item.get("expected_pages", [])}
    expected.discard(None)
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(
            f"{base_url.rstrip('/')}/api/reports/{requests.utils.quote(str(item['report_id']), safe='')}/ask",
            json={"question": item["question"], "mode": mode},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        evidence_pages: list[int] = []
        for chunk in payload.get("evidence", []) or []:
            pages = [page_number(str(value)) for value in chunk.get("evidence", [])]
            evidence_pages.extend(page for page in pages if page is not None)
        hit_ranks = [index for index, page in enumerate(evidence_pages, start=1) if page in expected]
        answer = str(payload.get("answer") or "")
        answer_norm = normalized(answer)
        required = [normalized(str(value)) for value in item.get("required_terms", [])]
        forbidden = [normalized(str(value)) for value in item.get("forbidden_terms", [])]
        required_recall = (
            sum(term in answer_norm for term in required) / len(required) if required and mode == "live" else None
        )
        forbidden_pass = not any(term in answer_norm for term in forbidden) if forbidden and mode == "live" else None
        abstention_pass = None
        if mode == "live" and item.get("expect_abstain"):
            abstention_pass = any(marker in answer_norm for marker in ABSTAIN_MARKERS)
        return CaseResult(
            id=str(item["id"]), report_id=str(item["report_id"]), status="passed",
            retrieval_hit=bool(hit_ranks), reciprocal_rank=(1 / min(hit_ranks)) if hit_ranks else 0.0,
            required_term_recall=required_recall, forbidden_term_pass=forbidden_pass,
            abstention_pass=abstention_pass,
            citation_qc=(payload.get("citation_qc") or {}).get("status"),
            latency_ms=round((time.perf_counter() - started) * 1000), evidence_pages=evidence_pages,
            answer=answer if mode == "live" else None, error=None,
        )
    except Exception as error:
        return CaseResult(
            id=str(item["id"]), report_id=str(item["report_id"]), status="error",
            retrieval_hit=False, reciprocal_rank=0.0, required_term_recall=None,
            forbidden_term_pass=None, abstention_pass=None, citation_qc=None,
            latency_ms=round((time.perf_counter() - started) * 1000), evidence_pages=[],
            answer=None, error=f"{type(error).__name__}: {error}",
        )


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    successful = [result for result in results if result.status == "passed"]
    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None
    def percentile(values: list[int], fraction: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return ordered[index]
    latencies = [result.latency_ms for result in successful]
    return {
        "cases": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "hit_at_5": average([float(result.retrieval_hit) for result in successful]),
        "mrr": average([result.reciprocal_rank for result in successful]),
        "effective_hit_at_5": average([float(result.retrieval_hit) for result in results]),
        "effective_mrr": average([result.reciprocal_rank for result in results]),
        "required_term_recall": average([result.required_term_recall for result in successful if result.required_term_recall is not None]),
        "citation_pass_rate": average([float(result.citation_qc == "pass") for result in successful if result.citation_qc]),
        "mean_latency_ms": average([float(result.latency_ms) for result in successful]),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
    }


def markdown_report(run: dict[str, Any]) -> str:
    metrics = run["metrics"]
    lines = ["# RAG regression report", "", f"Generated: {run['generated_at']}", f"Mode: `{run['mode']}`", "", "## Metrics", ""]
    lines.extend(f"- {key}: **{value}**" for key, value in metrics.items())
    lines.extend(["", "## Failed or missed cases", ""])
    misses = [item for item in run["results"] if item["status"] != "passed" or not item["retrieval_hit"]]
    if not misses:
        lines.append("No misses.")
    for item in misses:
        lines.append(f"- `{item['id']}`: {item.get('error') or 'expected page absent from retrieval'}")
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, generated_at: str, mode: str, results: list[CaseResult]) -> None:
    run = {
        "schema_version": 1,
        "generated_at": generated_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "metrics": aggregate(results),
        "results": [asdict(result) for result in results],
    }
    temporary = output_dir / "report.json.part"
    temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "report.json")
    (output_dir / "report.md").write_text(markdown_report(run), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoArchiveAI RAGFlow regression")
    parser.add_argument("--gold", type=Path, default=Path("evaluation/rag/gold.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", choices=("retrieval_only", "live"), default="retrieval_only")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="RAGFlow concurrency. Keep 1 for release baselines; higher values are load tests.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    items = gold.get("items", [])
    if not args.include_candidates:
        items = [item for item in items if item.get("review_status") == "approved"]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No approved cases. Review gold.json or pass --include-candidates.")
    generated_at = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir or Path("results/rag_evaluation") / generated_at.replace(":", "-")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    report_path = output_dir / "report.json"
    if args.resume and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        generated_at = previous.get("generated_at", generated_at)
        results = [CaseResult(**item) for item in previous.get("results", [])]
    completed_ids = {result.id for result in results}
    pending = [item for item in items if str(item["id"]) not in completed_ids]
    if args.workers == 1:
        for item in pending:
            results.append(evaluate_case(args.base_url, item, args.mode, args.timeout))
            write_outputs(output_dir, generated_at, args.mode, results)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            results.extend(
                executor.map(
                    lambda item: evaluate_case(args.base_url, item, args.mode, args.timeout),
                    pending,
                )
            )
        write_outputs(output_dir, generated_at, args.mode, results)
    write_outputs(output_dir, generated_at, args.mode, results)
    print(json.dumps(aggregate(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
