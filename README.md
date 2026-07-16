# GeoArchiveAI

Сервис обработки архивных геолого-геофизических отчётов: регистрация фонда,
классификация страниц, GPU OCR, загрузка в RAGFlow, извлечение сущностей агентами
и подготовка картографических артефактов.

## Быстрый запуск

```powershell
scripts\start_geoarchive.cmd
```

После запуска:

- интерфейс: http://127.0.0.1:8501
- Swagger: http://127.0.0.1:8765/docs
- OpenAPI: http://127.0.0.1:8765/openapi.json
- состояние сервисов: http://127.0.0.1:8765/api/services
- состояние автоматической очереди: http://127.0.0.1:8765/api/queue

Каталог `reports_inbox/` отслеживается автоматически. После завершения копирования
отчёт регистрируется и ставится в обработку без отдельной кнопки в интерфейсе.
По умолчанию одновременно обрабатываются не более двух отчётов; лимит задаётся
переменной `AUTO_MAX_REPORTS`.

## Структура

```text
apps/
  api/                 FastAPI control plane и тесты API
  web/                 тонкий Streamlit-клиент
pipeline/              активный конвейер обработки отчёта
services/
  ocr/                 GPU PaddleOCR HTTP-сервис
  map_digitizer/       векторизация карт и контракт map-агента
research/              старые эксперименты; продуктом не вызываются
scripts/               команды запуска локального стенда
docs/                  архитектура и материалы конференции
reports_inbox/         входные данные, не хранятся в Git
reports_staging/       распакованные архивы, не хранятся в Git
runs/                  состояния и результаты запусков, не хранятся в Git
secrets/               локальные ключи, не хранятся в Git
```

## Поток данных

```text
folder/upload -> page routing -> PaddleOCR -> Markdown -> RAGFlow -> search
                           map pages -> map digitizer -> source/digitized pair
```

FastAPI отвечает за загрузку, регистрацию, запуск и статус задач. Streamlit не
запускает скрипты и не читает `runs/` напрямую. PaddleOCR работает отдельным
GPU-сервисом, а RAGFlow остаётся отдельным хранилищем и retrieval-сервисом.

## Основные API

- `POST /api/reports/upload` — загрузить файл и зарегистрировать отчёты.
- `POST /api/intake/scan` — зарегистрировать уже скопированные фонды.
- `GET /api/reports` — список отчётов и состояния.
- `POST /api/reports/{report_id}/run` — запустить или повторить обработку.
- `GET /api/reports/{report_id}/status` — стадии и ошибки.
- `GET /api/reports/{report_id}/maps` — только исходные и оцифрованные карты.
- `GET /api/reports/{report_id}` — итоговый bundle.
- `POST /api/reports/{report_id}/ask` — вопрос по обработанному отчёту.
- `POST /api/search` — RAGFlow-поиск и единый ответ по всему фонду.

Полные схемы запросов и ответов всегда доступны в Swagger.

## Экспорт поверхности в Petrel

Оцифрованная карта экспортируется как CPS-3 ASCII Grid. Для проекта в
`ГК-42 21N` используется файл `*_gk42_21n.cps3` и система
`Pulkovo 1942 / Gauss-Kruger 21N` (`EPSG:28481`, современный эквивалент
`EPSG:2511`). Координаты X/Y и значения Z записаны в метрах. Не следует выбирать
`EPSG:28421`: это вариант с номером зоны в координате X, где значения начинаются
примерно с 21 000 000 м.

Архивная карта хранится в `ГК-42 19N`; файл `*_gk42_19n.cps3` оставлен как
контрольный исходный вариант. В проект 21N его можно загружать только после
перепроецирования. Текущий результат имеет статус `REVIEW`, пока профильная
геопривязка не подтверждена дополнительной контрольной линией.

## RAG benchmark

Эталонные вопросы хранятся в `benchmarks/rag/`. Проверка разделяет потери по
слоям: наличие страницы в OCR Markdown, сохранность ключевых фактов и чисел,
позиция эталонной страницы в выдаче RAGFlow и сетевые ошибки retrieval.

```powershell
$env:PYTHONPATH="apps/api;."
python apps/api/rag_benchmark.py `
  benchmarks/rag/384092.json `
  runs/384092/report_384092_fast_ocr.md `
  runs/384092/ragflow.json `
  secrets/ragflow_token.txt `
  --output runs/384092/rag_benchmark.json `
  --manifest runs/384092/job.json `
  --limit 10
```

Результаты сохраняются одновременно в JSON и Markdown. Базовые целевые пороги
для MVP: OCR page/fact recall не ниже 95%, retrieval Hit@5 не ниже 90% на
доступных OCR-фактах, отсутствие timeout/error. Новые отчёты и старые фонды при
повторном запуске ставят в OCR все текстовые страницы и оглавления.

## Разработка

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s pipeline -p "test_*.py" -q
python -m unittest discover -s apps/api -p "test_*.py" -q
python -m unittest discover -s services/map_digitizer/tests -p "test_*.py" -q
python -m unittest discover -s services/ocr -p "test_*.py" -q
python -m unittest apps.web.test_media -q
```

Рабочие изменения фиксируются отдельными коммитами. Большие отчёты, OCR-вывод,
модели и временные изображения исключены через `.gitignore`.
