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
