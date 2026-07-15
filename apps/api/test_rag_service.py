from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rag_service import (
    CorpusAnswerService,
    CorpusLocalRetriever,
    CorpusRagflowRetriever,
    LocalCorpusRetriever,
    RagflowClient,
    ReportAnswerService,
    ReportConfig,
    cited_page_numbers,
    page_ids,
    retrieval_source,
    report_id_from_document,
)


class FakeRagflow:
    def retrieve(self, question: str, limit: int = 5):
        return {
            "latency_ms": 12,
            "transport": "ragflow",
            "chunks": [
                {
                    "rank": 1,
                    "similarity": 0.75,
                    "document_name": "report.md",
                    "evidence": ["text:00014"],
                    "excerpt": "Работы выполнялись по горизонтам К и КВ.",
                    "raw_content": "[SOURCE_PAGE: page_00014.jpg] Работы выполнялись по горизонтам К и КВ.",
                }
            ],
        }


class FakeCompletions:
    def __init__(self, answer: str):
        self.answer = answer
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))]
        )


class FakeLlm:
    def __init__(self, answer: str):
        self.chat = SimpleNamespace(completions=FakeCompletions(answer))


class FakeSession:
    def __init__(self):
        self.request_json = None

    def post(self, _url, **kwargs):
        self.request_json = kwargs["json"]
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"code": 0, "data": {"chunks": [], "total": 0}},
        )


class RagServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        bundle = {"preset_queries": []}
        metadata = {"document_state": {"dataset_id": "dataset-1"}}
        (root / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (root / "corpus.md").write_text(
            "[SOURCE_PAGE: page_00014.jpg] Изучались горизонты К и КВ.\n"
            "[SOURCE_PAGE: page_00025.jpg] Скважина 410 имела газопроявления.",
            encoding="utf-8",
        )
        (root / "rag.token").write_text("token=test", encoding="utf-8")
        (root / "llm.env").write_text(
            "OPENAI_API_KEY=test\nBASE_URL=http://localhost.invalid/v1", encoding="utf-8"
        )
        self.config = ReportConfig(
            report_id="test",
            bundle_path=root / "bundle.json",
            ragflow_metadata_path=root / "metadata.json",
            local_corpus_path=root / "corpus.md",
            ragflow_token_path=root / "rag.token",
            llm_credentials_path=root / "llm.env",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_page_ids_are_deduplicated(self):
        content = "[SOURCE_PAGE: page_00014.jpg] a\n[SOURCE_PAGE: page_00014.jpg] b"
        self.assertEqual(page_ids(content), ["text:00014"])

    def test_current_source_page_marker_is_supported(self):
        content = "[SOURCE_PAGE: page:00014; path=scan.tif] text"
        self.assertEqual(page_ids(content), ["text:00014"])

    def test_ragflow_retrieval_is_scoped_to_report_documents(self):
        session = FakeSession()
        client = RagflowClient(
            token="token",
            dataset_id="dataset",
            document_ids=["doc-1", "doc-2"],
            proxy_url=None,
            session=session,
        )

        client.retrieve("question")

        self.assertEqual(session.request_json["dataset_ids"], ["dataset"])
        self.assertEqual(session.request_json["document_ids"], ["doc-1", "doc-2"])

    def test_dataset_retrieval_omits_empty_document_filter(self):
        session = FakeSession()
        client = RagflowClient(
            token="token", dataset_id="dataset", document_ids=[], proxy_url=None, session=session
        )

        client.retrieve("question")

        self.assertNotIn("document_ids", session.request_json)

    def test_report_id_is_read_from_ragflow_document_name(self):
        name = "report_377069_fast_ocr_part_002_0fe803101a03.md"
        self.assertEqual(report_id_from_document(name), "377069")

    def test_corpus_retriever_merges_and_reranks_reports(self):
        first = FakeRagflow()
        second = FakeRagflow()
        second.retrieve = lambda question, limit=5: {
            **FakeRagflow().retrieve(question, limit),
            "chunks": [{
                **FakeRagflow().retrieve(question, limit)["chunks"][0],
                "similarity": 0.9,
                "document_name": "report_377069_fast_ocr_part_001_deadbeef.md",
            }],
        }

        result = CorpusRagflowRetriever([first, second], workers=2).retrieve("приток", limit=2)

        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(result["chunks"][0]["document_name"], "report_377069_fast_ocr_part_001_deadbeef.md")
        self.assertEqual(result["transport"], "ragflow_fanout")

    def test_local_corpus_search_preserves_report_identity(self):
        second_root = Path(self.temp.name) / "second"
        second_root.mkdir()
        second_corpus = second_root / "corpus.md"
        second_corpus.write_text(
            "[SOURCE_PAGE: page:00031; path=scan.tif] Получен приток нефти из скважины 12.",
            encoding="utf-8",
        )
        second = ReportConfig(
            report_id="377069",
            bundle_path=self.config.bundle_path,
            ragflow_metadata_path=self.config.ragflow_metadata_path,
            local_corpus_path=second_corpus,
            ragflow_token_path=self.config.ragflow_token_path,
            llm_credentials_path=self.config.llm_credentials_path,
        )

        result = CorpusLocalRetriever([self.config, second]).retrieve("приток нефти", limit=5)

        self.assertEqual(result["chunks"][0]["report_id"], "377069")
        self.assertEqual(result["chunks"][0]["evidence"], ["text:00031"])

    def test_grouped_citations_are_parsed(self):
        self.assertEqual(cited_page_numbers("Ответ [стр. 3, 154; 115]."), {3, 115, 154})

    def test_proxy_transport_is_reported_as_ragflow(self):
        self.assertEqual(retrieval_source("proxy", "retrieval"), "ragflow_retrieval")

    def test_local_retrieval_matches_russian_inflection(self):
        retriever = LocalCorpusRetriever(self.config.local_corpus_path)
        result = retriever.retrieve("Какие горизонты изучались?")
        self.assertEqual(result["chunks"][0]["evidence"], ["text:00014"])

    def test_local_retrieval_reads_current_markdown_marker(self):
        self.config.local_corpus_path.write_text(
            "[SOURCE_PAGE: page:00014; path=scan.tif] Скважина 410 испытана.",
            encoding="utf-8",
        )
        retriever = LocalCorpusRetriever(self.config.local_corpus_path)

        result = retriever.retrieve("Какая скважина испытана?")

        self.assertEqual(result["chunks"][0]["evidence"], ["text:00014"])

    def test_valid_citation_passes_qc(self):
        service = ReportAnswerService(
            self.config,
            ragflow=FakeRagflow(),
            llm=FakeLlm("Работы выполнялись по горизонтам К и КВ [стр. 14]."),
        )
        result = service.ask("Какие горизонты изучались?")
        self.assertEqual(result["citation_qc"]["status"], "pass")
        self.assertEqual(result["citation_qc"]["cited_pages"], [14])

    def test_retrieval_only_reports_transport_source(self):
        service = ReportAnswerService(
            self.config,
            ragflow=FakeRagflow(),
            llm=FakeLlm("unused"),
        )
        result = service.ask("Какие горизонты изучались?", allow_llm=False)
        self.assertEqual(result["source"], "ragflow_retrieval")
        self.assertEqual(result["retrieval_transport"], "ragflow")

    def test_llm_answer_is_cached(self):
        llm = FakeLlm("Ответ [стр. 14].")
        service = ReportAnswerService(self.config, ragflow=FakeRagflow(), llm=llm)
        first = service.ask("Какие горизонты изучались?")
        second = service.ask("Какие горизонты изучались?")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(llm.chat.completions.call_count, 1)

    def test_hallucinated_citation_is_flagged(self):
        service = ReportAnswerService(
            self.config,
            ragflow=FakeRagflow(),
            llm=FakeLlm("Ответ [стр. 99]."),
        )
        result = service.ask("Какие горизонты изучались?")
        self.assertEqual(result["citation_qc"]["status"], "review")
        self.assertEqual(result["citation_qc"]["invalid"], [99])

    def test_missing_citation_is_flagged(self):
        service = ReportAnswerService(
            self.config,
            ragflow=FakeRagflow(),
            llm=FakeLlm("Работы выполнялись по горизонтам К и КВ."),
        )
        result = service.ask("Какие горизонты изучались?")
        self.assertTrue(result["citation_qc"]["missing_citations"])

    def test_corpus_answer_exposes_report_id_and_source_citation(self):
        service = CorpusAnswerService(
            self.config,
            ragflow=FakeRagflow(),
            llm=FakeLlm("В отчёте 377069 отмечены газопроявления [источник 1]."),
        )
        service.ragflow.retrieve = lambda question, limit=12: {
            **FakeRagflow().retrieve(question, limit),
            "chunks": [{
                **FakeRagflow().retrieve(question, limit)["chunks"][0],
                "document_name": "report_377069_fast_ocr_part_001_deadbeef.md",
            }],
        }

        result = service.ask("Где были приточные скважины?")

        self.assertEqual(result["evidence"][0]["report_id"], "377069")
        self.assertEqual(result["citation_qc"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
