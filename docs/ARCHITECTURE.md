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
Streamlit UI
```

FastAPI является единственным управляющим слоем. Streamlit показывает данные
API и не запускает обработку самостоятельно. PaddleOCR и RAGFlow являются
внешними сервисами. Состояние незавершённых задач хранится в `runs/`.

## Точки входа

- `apps/api/app/main.py` — HTTP-приложение;
- `pipeline/job_worker.py` — полный отчёт;
- `services/map_digitizer/pipeline.py` — одна карта;
- `services/ocr/app.py` — GPU OCR API.
