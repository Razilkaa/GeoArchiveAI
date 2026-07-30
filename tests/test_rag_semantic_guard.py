import json
from pathlib import Path
from types import SimpleNamespace

from rag_service import ReportAnswerService, ReportConfig


class FakeRagflow:
    def retrieve(self, question: str) -> dict:
        return {
            "chunks": [
                {
                    "rank": 1,
                    "similarity": 0.9,
                    "document_name": "report.md",
                    "evidence": ["text:00112"],
                    "excerpt": (
                        "Передать под глубокое поисковое бурение Варваровскую структуру. "
                        "Провести детальные работы на Западно-Сергеевской и Барановской."
                    ),
                    "raw_content": "source",
                }
            ],
            "latency_ms": 10,
            "transport": "direct",
        }


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        content = (
            "Для глубокого бурения рекомендованы Варваровская, "
            "Западно-Сергеевская и Барановская структуры [стр. 112]."
            if self.calls == 1
            else "Под глубокое бурение передаётся только Варваровская структура [стр. 112]."
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_relation_guard_rewrites_conflated_recommendations(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    metadata = tmp_path / "ragflow.json"
    corpus = tmp_path / "report.md"
    token = tmp_path / "token.txt"
    credentials = tmp_path / "credentials.txt"
    bundle.write_text(
        json.dumps(
            {
                "entities": {
                    "structures": [
                        {
                            "name": "Варваровская структура",
                            "status": "подготовлена под глубокое бурение",
                            "evidence": ["page:00112"],
                        },
                        {
                            "name": "Западно-Сергеевская структура",
                            "status": "нужны детальные работы",
                            "evidence": ["page:00112"],
                        },
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metadata.write_text('{"dataset_id":"dataset","document_id":"document"}', encoding="utf-8")
    corpus.write_text("report", encoding="utf-8")
    token.write_text("token", encoding="utf-8")
    credentials.write_text("OPENAI_API_KEY=test", encoding="utf-8")
    config = ReportConfig(
        report_id="demo",
        bundle_path=bundle,
        ragflow_metadata_path=metadata,
        local_corpus_path=corpus,
        ragflow_token_path=token,
        llm_credentials_path=credentials,
    )
    completions = FakeCompletions()
    llm = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    service = ReportAnswerService(
        config,
        ragflow=FakeRagflow(),
        llm=llm,
    )

    result = service.ask("Какие структуры рекомендованы для глубокого бурения?")

    assert completions.calls == 2
    assert result["generation_attempts"] == 2
    assert result["answer"] == (
        "Под глубокое бурение передаётся только Варваровская структура [стр. 112]."
    )
