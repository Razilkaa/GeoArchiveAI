from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_benchmark import aggregate, evaluate_case, markdown_pages


class RagBenchmarkTest(unittest.TestCase):
    def test_distinguishes_missing_ocr_page_from_retrieval_miss(self):
        case = {
            "id": "well",
            "question": "Где скважина?",
            "expected_pages": [12],
            "term_groups": [["Р-410"], ["4505"]],
        }
        missing = evaluate_case(case, {}, {"chunks": []})
        self.assertEqual(missing["diagnosis"], "ocr_page_missing")

        pages = {12: "Скважина Р-410 закончена на глубине 4505 м"}
        retrieval_miss = evaluate_case(case, pages, {"chunks": []})
        self.assertEqual(retrieval_miss["diagnosis"], "retrieval_miss")

    def test_scores_relevant_page_rank_and_terms(self):
        case = {
            "id": "well",
            "question": "Где скважина?",
            "expected_pages": [12],
            "term_groups": [["Р-410"], ["4505"]],
        }
        retrieved = {
            "latency_ms": 10,
            "chunks": [
                {"rank": 1, "evidence": ["text:00003"], "excerpt": "другое"},
                {"rank": 2, "evidence": ["text:00012"], "excerpt": "Р-410 4505 м"},
            ],
        }
        result = evaluate_case(case, {12: "Р-410 4505 м"}, retrieved)
        self.assertEqual(result["diagnosis"], "pass")
        self.assertEqual(result["retrieval"]["first_relevant_rank"], 2)
        self.assertEqual(result["retrieval"]["reciprocal_rank"], 0.5)

    def test_retrieval_error_does_not_hide_available_ocr(self):
        case = {
            "id": "well",
            "question": "Где скважина?",
            "expected_pages": [12],
            "term_groups": [["Р-410"]],
        }
        result = evaluate_case(
            case,
            {12: "Скважина Р-410"},
            {"chunks": [], "error": "ReadTimeout"},
        )
        self.assertEqual(result["diagnosis"], "retrieval_error")
        self.assertEqual(result["retrieval"]["error"], "ReadTimeout")

    def test_reads_current_source_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "report.md"
            source.write_text(
                "[SOURCE_PAGE: page:00012; path=scan.jpg] Первая строка\n"
                "[SOURCE_PAGE: page:00012; path=scan.jpg] Вторая строка\n",
                encoding="utf-8",
            )
            pages = markdown_pages(source)
        self.assertIn("Первая строка", pages[12])
        self.assertIn("Вторая строка", pages[12])

    def test_aggregate_reports_layer_metrics(self):
        results = [
            {"diagnosis": "pass", "ocr": {"page_recall": 1.0, "term_recall": 1.0}, "retrieval": {"hit_at_5": True, "reciprocal_rank": 1.0, "term_recall": 1.0}},
            {"diagnosis": "ocr_page_missing", "ocr": {"page_recall": 0.0, "term_recall": 0.0}, "retrieval": {"hit_at_5": False, "reciprocal_rank": 0.0, "term_recall": 0.0}},
        ]
        summary = aggregate(results)
        self.assertEqual(summary["ocr_page_recall"], 0.5)
        self.assertEqual(summary["retrieval_eligible_cases"], 1)
        self.assertEqual(summary["retrieval_hit_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
