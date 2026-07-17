# Map digitizer

Детерминированный конвейер оцифровки структурных карт. LLM не участвует в
геометрии: PaddleOCR читает подписи, OpenCV выделяет линии, численный модуль
строит топологически корректную поверхность.

## Запуск

Сервис PaddleOCR должен отвечать на `http://127.0.0.1:18080`.

```powershell
python -m services.map_digitizer.pipeline "path/to/map.jpg" `
  --output-dir "runs/maps/example"
```

Необязательные параметры:

- `--interval 0.1` фиксирует сечение в километрах; без него сечение определяется автоматически.
- `--trace-scale 0.6` задаёт рабочее разрешение трассировщика.
- `--ocr-api-url http://host:18080` переопределяет адрес GPU OCR.

Повторный запуск использует сохранённый `ocr.json`, поэтому OCR не оплачивается
и не выполняется заново.

Пакет папок и файлов запускается последовательно и возобновляется после сбоя:

```powershell
python -m services.map_digitizer.batch "reports/maps" `
  --output-dir "runs/map_batch"
```

Сводный benchmark по готовым manifest:

```powershell
python -m services.map_digitizer.benchmark "runs/map_batch" `
  --output "runs/map_batch/benchmark.json"
```

## Этапы

1. `ocr` читает текст и координаты подписей.
2. `trace` отделяет профили, текст и изолинии и сшивает разрывы.
3. `assignment` определяет сечение, присваивает значения и фильтрует OCR-выбросы.
4. `reconstruction` автоматически выбирает режим:
   - `dense_profile_measurements` для карт с плотными отметками на профилях;
   - `sparse_labels_trace_guided` для карт только с подписями изолиний.
5. `quality` проверяет ошибки ограничений, пересечения, диапазон и лишние замыкания.

## Результат

Главный контракт находится в `pipeline_result.json`:

- `status`: `accepted`, `review` или `failed`;
- `stages`: время и метрики каждого этапа;
- `quality.reasons`: причины ручной или модельной проверки;
- `artifacts.surface_preview`: сравнение исходника и результата;
- `artifacts.pixel_grid`: сетка в пиксельной системе до геопривязки;
- `artifacts.pixel_contours`: изолинии GeoJSON в пикселях.

Каждая ошибка сохраняется вместе с `failed_stage`; сбой одной страницы не должен
останавливать пакет отчётов. CPS-3 экспорт выполняется только после определения
аффинного преобразования и системы координат проекта.

## Проверка

```powershell
python -m unittest discover -s services/map_digitizer/tests -q
```

Исследовательские прототипы и тяжёлые промежуточные файлы остаются в
`research/map_digitization/` и `pipeline/tracing/`; production-код находится
только в `services/map_digitizer/`.
