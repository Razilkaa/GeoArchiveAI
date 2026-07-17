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
   - `dense_profile_measurements` для карт минимум с 80 пригодными отметками на
     профилях;
   - `sparse_labels_trace_guided` для карт только с подписями изолиний.
5. `quality` проверяет ошибки ограничений, пересечения, диапазон и лишние замыкания.

Sparse reconstruction is deliberately always marked `review`: a deterministic
line tracer cannot prove that several contour families on one sheet belong to
the same geological horizon. Dense outputs can be auto-accepted only after the
remaining numerical and topology checks pass. Pipeline version changes
invalidate geometry results automatically while retaining the expensive OCR
cache.

Large scans are downscaled to a 4000-pixel maximum side before OCR when they
exceed 40 megapixels. Coordinates are restored to the source image; overlapping
tiles are used only as an OOM fallback. This bounds GPU memory without four
expensive requests per normal 8K scan. Reconstructed
surfaces reject short unlabelled trace fragments and isolated closed contours
without source or adjacent-level support.

`services.map_digitizer.report_batch` consumes the report `job.json`, processes
pages routed as `map` or `chart`, rejects visual duplicates and writes the
existing `map_agent/result.json` contract. `chart` candidates can never be
auto-accepted. The report worker runs this stage automatically after page
routing and preserves external/manual georeferenced artifacts on reruns.

## Результат

Главный контракт находится в `pipeline_result.json`:

- `status`: `accepted`, `review`, `not_applicable` или `failed`;
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

## Georeferencing and API

The pixel grid is georeferenced only after at least three pixel-to-map control
points and the project CRS are known. Four or more controls provide an
independent residual check; a three-point affine fit is always marked `review`.

```powershell
python -m services.map_digitizer.georeference_grid `
  "runs/maps/example/surface_grid_pixel.npz" "control_points.json" `
  --output-dir "runs/maps/example/georeferenced" `
  --target-crs "EPSG:28421" --cell-size 25 --name horizon_k
```

`control_points.json` is an array of objects with `pixel: [x, y]` and
`map: [easting, northing]`. The command exports CPS-3, XYZ, PRJ and a QC JSON.

The same workflow is exposed by FastAPI:

- `POST /api/maps/jobs` uploads an image and starts digitization;
- `GET /api/maps/jobs/{job_id}` returns stage, quality and artifact metadata;
- `GET /api/maps/jobs/{job_id}/artifacts/{artifact_name}` downloads a generated
  preview, pixel grid, GeoJSON, CPS-3, XYZ or PRJ file;
- `POST /api/maps/jobs/{job_id}/georeference` exports a CRS-aware grid.
- `POST /api/reports/{report_id}/maps/run` reruns map discovery and digitization
  for an already registered report, then rebuilds its result bundle.

Artifact names are stable API identifiers: `surface_preview`, `pixel_grid`,
`pixel_contours`, `cps3`, `xyz`, `prj` and `georeference_metadata`. The API never
serves arbitrary paths from a job manifest.

Report map metadata includes an `artifact_id` and `download_url` for each source
and generated artifact. `GET /api/reports/{report_id}/maps/artifacts/{artifact_id}`
serves only files under that report run or its registered source directory.
Standalone jobs are capped by `MAP_MAX_JOBS` (default: 2); excess submissions
receive HTTP 429 instead of oversubscribing the OCR GPU.

Run the API from `apps/api` and inspect the contract at `/docs`.

Исследовательские прототипы и тяжёлые промежуточные файлы остаются в
`research/map_digitization/` и `pipeline/tracing/`; production-код находится
только в `services/map_digitizer/`.
