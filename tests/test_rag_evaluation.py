from evaluation.rag.run_regression import CaseResult, aggregate, page_number


def test_page_number_accepts_api_and_bundle_identifiers() -> None:
    assert page_number("page:00156") == 156
    assert page_number("text:00003") == 3
    assert page_number("unknown") is None


def test_aggregate_keeps_retrieval_separate_from_answer_metrics() -> None:
    results = [
        CaseResult("a", "1", "passed", True, 1.0, None, None, None, "pass", 100, [3], None, None),
        CaseResult("b", "1", "passed", False, 0.0, None, None, None, "pass", 300, [4], None, None),
    ]
    metrics = aggregate(results)
    assert metrics["hit_at_5"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["required_term_recall"] is None
    assert metrics["mean_latency_ms"] == 200.0
