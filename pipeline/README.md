# Active report pipeline

Здесь находится только код, вызываемый рабочим конвейером.

- `batch_intake.py` — регистрация входных фондов и создание manifest.
- `report_factory.py` — инвентаризация страниц.
- `page_router.py` — классификация text/graphic/map/table.
- `factory_runner.py` — GPU OCR, Markdown и RAGFlow ingest.
- `factory_agents.py` — извлечение структур, карт, результатов и скважин.
- `job_worker.py` — последовательность стадий и resume/retry.
- `report_orchestrator.py` — граф специализированных агентов.
- `build_result_bundle.py` — единый результат для API и UI.
- `watch_inbox.py` — необязательный watcher; основной путь идёт через FastAPI.

Одноразовые CV/VLM-эксперименты перенесены в `research/`.
