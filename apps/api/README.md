# FastAPI

Запуск из этой директории:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Swagger: http://127.0.0.1:8765/docs

- `app/main.py` создаёт приложение;
- `app/routers/` содержит HTTP endpoints;
- `app/services/` управляет задачами и результатами;
- `rag_service.py` выполняет retrieval только через RAGFlow и один LLM-вызов.

Локального BM25 и отдельного corpus index в приложении нет.
