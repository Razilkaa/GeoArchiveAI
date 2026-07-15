from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from corpus_search import CorpusSearchService, report_id_from_document
from rag_service import ReportConfig


class FakeRetriever:
    def __init__(self, document: str, similarity: float) -> None:
        self.document = document
        self.similarity = similarity

    def retrieve(self, question: str, limit: int = 5):
        return {
            "chunks": [{
                "rank": 1,
                "similarity": self.similarity,
                "document_name": self.document,
                "evidence": ["text:00031"],
                "excerpt": f"{question}: получен приток нефти.",
                "raw_content": "[SOURCE_PAGE: page:00031] получен приток нефти.",
            }],
            "latency_ms": 10,
            "transport": "ragflow",
        }


class FakeLlm:
    def __init__(self) -> None:
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="В отчёте 377069 получен приток [источник 1].")
        )])


class CorpusSearchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        for name, content in {
            "bundle.json": "{}",
            "metadata.json": json.dumps({"dataset_id": "dataset"}),
            "corpus.md": "",
            "token.txt": "token=test",
            "llm.env": "OPENAI_API_KEY=test\nBASE_URL=http://localhost.invalid/v1",
        }.items():
            (root / name).write_text(content, encoding="utf-8")
        self.config = ReportConfig(
            "test",
            root / "bundle.json",
            root / "metadata.json",
            root / "corpus.md",
            root / "token.txt",
            root / "llm.env",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_document_name_maps_to_report(self):
        name = "report_377069_fast_ocr_part_002_0fe803101a03.md"
        self.assertEqual(report_id_from_document(name), "377069")

    def test_top_chunks_are_merged_before_single_generation(self):
        llm = FakeLlm()
        service = CorpusSearchService(
            [self.config],
            retrievers=[
                FakeRetriever("report_375392_fast_ocr_part_001_aabb.md", 0.4),
                FakeRetriever("report_377069_fast_ocr_part_001_ccdd.md", 0.8),
            ],
            llm=llm,
            workers=2,
        )

        result = service.ask("Где были получены притоки?")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(result["evidence"][0]["report_id"], "377069")
        self.assertEqual(result["citation_qc"]["status"], "pass")

    def test_retrieval_only_does_not_call_llm(self):
        llm = FakeLlm()
        service = CorpusSearchService(
            [self.config],
            retrievers=[FakeRetriever("report_377069_fast_ocr_part_001_ccdd.md", 0.8)],
            llm=llm,
        )

        result = service.ask("Где были получены притоки?", allow_llm=False)

        self.assertEqual(llm.calls, 0)
        self.assertEqual(result["source"], "ragflow_retrieval")


if __name__ == "__main__":
    unittest.main()
