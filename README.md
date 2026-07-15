# GeoArchive AI

Рабочий прототип обработки архивных геолого-геофизических отчётов.

## Структура

- `384092/` — исходные материалы демонстрационного отчёта.
- `demo_app/` — пользовательский интерфейс и API поиска по отчёту.
- `pipeline/` — маршрутизация страниц, OCR, агенты и сборка результата.
- `reports_inbox/` — локальная очередь новых архивов.
- `reports_staging/` — распакованные и классифицированные материалы.
- `runs/` — воспроизводимые результаты запусков.
- `docs/` — тезисы и архитектурные документы.
- `archive/` — сохранённые исследовательские эксперименты, не входящие в продуктовый путь.
- `secrets/` — локальные ключи; каталог исключён из Git.

## Запуск

```powershell
cd C:\FINAM\Conference\demo_app\backend
python -m uvicorn api:app --host 127.0.0.1 --port 8765

cd C:\FINAM\Conference\demo_app
streamlit run app.py --server.port 8501
```

## Оператор отчёта

```powershell
python pipeline/report_orchestrator.py runs/384092/job.json plan
python pipeline/report_orchestrator.py runs/384092/job.json run
```

Оператор выполняет граф `classifier → (rag + map) → qc → export`, сохраняет checkpoint и возобновляет только изменившиеся ветви. Новые ZIP, TAR, RAR и 7z архивы регистрируются watcher-ом после стабилизации размера и освобождения файла процессом копирования.
