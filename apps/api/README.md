# FastAPI control plane

Запуск из этой директории:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

- `app/main.py` создаёт приложение и подключает routers.
- `app/routers/` содержит только HTTP-контракты.
- `app/services/` содержит регистрацию, jobs и чтение результатов.
- `app/schemas.py` содержит Pydantic-схемы.
- `rag_service.py` реализует вопросы по обработанному отчёту.
- `test_*.py` проверяют API и RAG-контракты.

Swagger: http://127.0.0.1:8765/docs
