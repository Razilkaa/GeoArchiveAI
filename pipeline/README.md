# Обработка отчёта

`job_worker.py` является единственным оркестратором полного отчёта.

```text
batch_intake -> page_router -> OCR/Markdown -> RAGFlow -> map digitizer -> bundle
```

Основные файлы:

- `batch_intake.py` регистрирует архив;
- `page_router.py` классифицирует страницы;
- `factory_runner.py` создаёт OCR Markdown и отправляет его в RAGFlow;
- `job_worker.py` последовательно запускает стадии и умеет продолжать задачу;
- `build_result_bundle.py` формирует ответ для API и UI;
- `factory_agents.py` и `report_orchestrator.py` извлекают сущности отчёта.

Картографический код здесь не находится. Карты всегда передаются в
`services/map_digitizer/pipeline.py`.
