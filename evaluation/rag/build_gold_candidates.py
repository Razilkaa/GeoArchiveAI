from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ENTITY_QUESTIONS = {
    "structures": "В отчёте упоминается структура «{label}»? Кратко опиши её статус.",
    "wells": "Что в отчёте сказано о скважине «{label}»?",
    "horizons": "Что в отчёте сказано о горизонте «{label}»?",
}


def label_for(entity: dict[str, Any]) -> str | None:
    for key in ("name", "id", "title"):
        value = entity.get(key)
        if value:
            return str(value).strip()
    return None


def candidates(runs_root: Path, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for bundle_path in sorted(runs_root.glob("*/result_bundle.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        report_id = str(bundle.get("report_id") or bundle_path.parent.name)
        entities = bundle.get("entities") or {}
        for kind, template in ENTITY_QUESTIONS.items():
            for entity in entities.get(kind, []) or []:
                label = label_for(entity)
                pages = [str(page) for page in entity.get("evidence", []) if page]
                if not label or not pages:
                    continue
                items.append(
                    {
                        "id": f"{report_id}:{kind}:{len(items) + 1:04d}",
                        "report_id": report_id,
                        "category": kind,
                        "question": template.format(label=label),
                        "expected_pages": pages,
                        "required_terms": [label],
                        "forbidden_terms": [],
                        "expect_abstain": False,
                        "review_status": "candidate",
                    }
                )
                if len(items) >= limit:
                    return items
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reviewable RAG gold-set draft")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/rag/gold.json"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "description": "GeoArchiveAI expert-review RAG regression set",
        "items": candidates(args.runs_root, max(1, args.limit)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['items'])} candidates to {args.output}")


if __name__ == "__main__":
    main()
