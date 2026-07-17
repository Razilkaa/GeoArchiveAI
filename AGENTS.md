# Навигатор по репозиторию (GeoArchiveAI)

Этот файл - карта репозитория для быстрой ориентации: что сейчас живое и рабочее,
что архив/эксперимент, что запускать и где лежат данные. Составлен 17.07.2026
по факту кода и git-истории (не по памяти/STATUS.md, которые местами устарели).

`STATUS.md` - исторический дневник ранней стадии проекта (детектор v4 + VLM,
14-15.07). Архитектура с тех пор сильно ушла вперёд. Актуальное описание системы -
`README.md` в корне, он поддерживается в актуальном состоянии.

## Как запустить всё

```
scripts\start_geoarchive.cmd   (или .ps1)
```

Поднимает стек: FastAPI на `http://127.0.0.1:8765` (Swagger `/docs`),
Streamlit-фронт на `http://127.0.0.1:8501`. Логи процессов - `runs/*.stdout.log`,
`runs/*.stderr.log`.

## Общая схема: что живое, что архив

| Директория | Статус | Роль |
|---|---|---|
| `apps/api` | ЖИВОЕ, основная точка разработки | FastAPI-бэкенд: приём отчётов, запуск обработки, RAG-поиск, карты |
| `apps/web` | ЖИВОЕ | Тонкий Streamlit-клиент, ходит в `apps/api`, сам ничего не считает |
| `pipeline/` | ЖИВОЕ, но стабильное (давно не менялось) | Обработка текста отчёта: интейк, OCR, RAGFlow, извлечение сущностей |
| `services/map_digitizer/` | ЖИВОЕ, сейчас основная точка разработки | Продакшн-пайплайн оцифровки карт (детерминированный CV, без LLM в геометрии) |
| `services/ocr/` | ЖИВОЕ | Отдельный GPU-сервис PaddleOCR, дергается и из `pipeline/`, и из `services/map_digitizer` |
| `research/map_digitization/` | R&D, НЕ вызывается продакшном | Старый детектор v4 + активный сейчас тред `tracing/` (оцифровка изогипс + геопривязка по профилям) |
| `research/legacy_pipeline/` | МЁРТВОЕ | Ранняя версия пайплайна, вытеснена `pipeline/` |
| `archive/` | МЁРТВОЕ, подтверждено git-историей | `map_experiments/`, `ui_experiment/` - тронуты только в первом коммите-бэйзлайне |
| `docs/legacy/` | МЁРТВОЕ, подтверждено git-историей | Старая документация, тронута только один раз при разделении product/research |

Правило проекта (см. `research/README.md`): если эксперимент в `research/`
"взлетает", его переносят в `services/` с контрактом и тестами. Пока он в
`research/`, продакшн его не вызывает.

## Точки входа

**Запустить обработку нового отчёта** - вручную запускать скрипты не нужно,
`apps/api` следит за `reports_inbox/` автоматически (`InboxMonitor` в
`apps/api/app/services/automation.py`). Ручные пути через API:
`POST /api/reports/upload` -> `POST /api/reports/{id}/run`.
Внутри цепочка стадий (`pipeline/job_worker.py`):
`batch_intake.py` -> `report_factory.py` -> `page_router.py` ->
`factory_runner.py` (GPU OCR/Markdown/RAGFlow) -> `factory_agents.py` ->
`report_orchestrator.py` -> `build_result_bundle.py`.
`pipeline/watch_inbox.py` - опциональный автономный вотчер вне FastAPI, для
обычной работы не нужен (см. `pipeline/README.md`).

**Оцифровать один скан карты** - продакшн-путь:
```
python -m services.map_digitizer.pipeline "path/to/map.jpg" --output-dir "runs/maps/example"
python -m services.map_digitizer.batch                      # пакетно
python -m services.map_digitizer.georeference_grid ... --target-crs EPSG:28421
```
То же самое доступно через API: `POST /api/maps/jobs`, `POST /api/maps/.../georeference`
(роутер `apps/api/app/routers/maps.py`). Скрипты в `research/map_digitization/tracing/`
в эту цепочку НЕ включены - это отдельный R&D-трек.

## `research/map_digitization/tracing/` - активный R&D по геопривязке

Это сейчас самый свежий тред (последний коммит на момент составления файла -
`2af3f12`, "research: preserve profile georeferencing prototype"). Задача -
геопривязка листа 23 через сеть сейсмопрофилей и мастер-шейп `shapes/srr_all.shp`,
отдельно от общей геопривязки в `services/map_digitizer`.

| Файл | Что делает |
|---|---|
| `trace_isolines.py` | v1-трассировщик изогипс на 1-битных сканах (фильтр по размеру компонент + Hough + обход скелета) |
| `crossing_filter.py` | Убирает пересекающиеся изогипсы одного горизонта (эвристика "вины" по прямизне/непараллельности) |
| `assign_values_23.py` | Прототип: присвоение значений трассированным изогипсам листа 23, поиск структур, проверка замыкания из текста отчёта |
| `surface_crosscheck_23.py` | Двухкарточная сверка: подписанные изогипсы vs поверхность по отметкам глубин, значения с флагом происхождения |
| `export_geojson_23.py` | Экспорт оцифрованных изогипс листа 23 в GeoJSON (пока в пиксельных координатах) |
| `detect_profile_lines.py` | Детекция прямых линий сейсмопрофилей на скане |
| `extract_profiles_shape.py` | Вырезает профили конкретного отчёта из мастер-шейпа `shapes/srr_all.shp` по инвентарному номеру |
| `plot_profiles_shape.py` | Визуальная сверка: профили из шейпа (реальные координаты) рядом с оцифрованными пиксельными линиями |
| `match_profile_labels.py` | Сопоставляет распознанные номера профилей с пиксельными линиями и профилями из шейпа |
| `profile_points_23.py` | Пересечение оцифрованных изогипс с линиями профилей -> точки (x, y, глубина), сверка с поверхностью |
| `georeference_profiles_23.py` | Предварительная геопривязка листа 23 через сеть профилей (least-squares) |

Рабочая цепочка этого треда: `extract_profiles_shape` -> `plot_profiles_shape`
(визуальная проверка) -> `match_profile_labels` -> `profile_points_23` ->
`crossing_filter` (чистка изогипс) -> `georeference_profiles_23`.

Важно: в `research/map_digitization/tracing/*.py` встречаются захардкоженные
пути вида `C:\FINAM\Conference\pipeline\tracing\out\23` - это артефакты из
времени, когда трассировка жила под `pipeline/`. Каталог `pipeline/tracing/`
всё ещё существует и содержит старые выходные данные (`batch_smoke/`,
`benchmarks/`, `e2e/`), но сами скрипты уже переехали в `research/`. Это
несостыковка путей, а не баг логики - при переносе скриптов стоит поправить.

## `runs/` - что реальное, что мусор

Папки с ID, совпадающим с `reports_inbox/` - реальные результаты обработки:
`363934, 365610, 368268, 368333, 370141, 375392, 375665, 377069, 378947,
379895, 380927, 384425, 393156`. В инбоксе есть ещё `364376, 378613, 385947` -
для них папок в `runs/` нет, похоже не обработаны.

`runs/384092` и `runs/384092_map_validation` - легитимны, это выходы по
эталонному вручную размеченному корпусу `384092/` (Хампинская с/п), не трогать.

Похоже на мусор/одноразовые тестовые прогоны (ID не из инбокса, нестандартные
имена): `runs/1`, `runs/2`, `runs/1174`, `runs/1174-2 ГР`, `runs/368268-PDF!`,
`runs/370141-PDF!`, `runs/CD отчет`. Ничего не удалено - решение оставлено за
пользователем.

Файлы прямо в `runs/` (`*.stdout.log`, `*.stderr.log`, `intake_summary.json`,
`intake_watcher.jsonl`) - служебные логи и состояние запущенного стека, не
результаты обработки отчётов. `runs/map_jobs`, `runs/map_validation`,
`runs/services` - служебные подпапки инструментов карт/бенчмарка.

## apps/api vs apps/web

`apps/api` - вся логика: приём и регистрация отчётов, запуск обработки,
RAG-поиск (RAGFlow), все эндпоинты по картам (`routers/maps.py`), плюс
`rag_benchmark.py`/`corpus_search.py`. Сейчас основная зона разработки, хорошо
покрыта тестами.

`apps/web` - тонкий Streamlit-клиент (`app.py`, `media.py`), сам ничего не
считает и не читает `runs/` напрямую, ходит в API по `BACKEND_URL =
"http://127.0.0.1:8765"`.

## Прочее

- `shapes/srr_all.*` - мастер-шейп сейсмопрофилей, используется только одним
  местом: `research/map_digitization/tracing/extract_profiles_shape.py`
  (фильтрация по `N_RGF`/`INV_svod`). В продакшн-геопривязке
  `services/map_digitizer` не участвует.
- `secrets/tokent.txt`, `secrets/ragflow_token.txt` - токены LLM/RAGFlow,
  используются в `apps/api/app/routers/reports.py`, `pipeline/job_worker.py`,
  `pipeline/watch_inbox.py` как дефолтные пути к файлам с credentials.
- `384092/` - единственный вручную размеченный эталонный корпус (Хампинская
  с/п, 1980). НЕ перезаписывать разметку без явного запроса.
- `benchmarks/maps/`, `benchmarks/rag/` - наборы для регрессионных проверок
  качества, см. `docs/MAP_DIGITIZATION_BENCHMARK.md` (актуален на 17.07,
  38 сканов).

## Актуальные README для деталей

- `README.md` - главное описание системы, поток данных, API, актуален.
- `pipeline/README.md` - что реально вызывается пайплайном текста/OCR.
- `research/README.md` - правило "эксперимент взлетел -> переезжает в services/".
- `services/map_digitizer/README.md` - CLI продакшн-пайплайна карт.
- `docs/architecture/AGENT_PIPELINE.md` - концептуальная схема 5 агентов.
- `docs/MAP_DIGITIZATION_BENCHMARK.md` - методика и результаты бенчмарка карт.
