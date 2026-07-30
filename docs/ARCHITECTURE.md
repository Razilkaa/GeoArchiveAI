# Архитектура

```text
reports_inbox
      |
      v
FastAPI -> report worker -> PaddleOCR -> Markdown -> RAGFlow
                         |
                         +-> map digitizer -> PNG / GeoJSON / CPS-3
      |
      v
React UI / compatibility Streamlit client
```

FastAPI является единственным управляющим слоем. Streamlit показывает данные
API и не запускает обработку самостоятельно. PaddleOCR и RAGFlow являются
внешними сервисами. Состояние незавершённых задач хранится в `runs/`.

## Production deployment

На сервере React отдаётся Nginx. FastAPI не запускает полный отчёт внутри
своего процесса: запрос помещается в `runs/_queue`, а отдельный worker вызывает
единственный production-оркестратор `pipeline/job_worker.py`. API и worker
используют общий persistent volume. PaddleOCR доступен по имени compose-сервиса,
RAGFlow остаётся единственным retrieval-слоем.

Все пути, URL, лимиты, модели и параметры retrieval задаются через
`geoarchive.settings` и переменные окружения. Секреты монтируются read-only и
не входят в образы.

## Точки входа

- `apps/api/app/main.py` — HTTP-приложение;
- `pipeline/job_worker.py` — полный отчёт;
- `services/map_digitizer/pipeline.py` — одна карта;
- `services/ocr/app.py` — GPU OCR API.
