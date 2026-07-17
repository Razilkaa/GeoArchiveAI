# GeoArchiveAI

Сервис принимает архив геолого-геофизического отчёта и автоматически создаёт:

- OCR Markdown для загрузки в RAGFlow;
- поиск и ответы по отчётам;
- список карт отчёта;
- маски изолиний и сейсмических профилей;
- оцифрованную карту, GeoJSON и CPS-3 Grid.

## Запуск

```powershell
scripts\start_geoarchive.cmd
```

- UI: http://127.0.0.1:8501
- Swagger: http://127.0.0.1:8765/docs

Новые отчёты кладутся в `reports_inbox/` и регистрируются автоматически.

## Где что находится

```text
apps/api/                 FastAPI и интеграция с RAGFlow
apps/web/                 Streamlit UI
pipeline/                 обработка полного отчёта
services/ocr/             GPU PaddleOCR API
services/map_digitizer/   единственный production-пайплайн карт
results/                  понятные итоговые файлы для просмотра и выгрузки
runs/                     внутреннее состояние задач; руками не редактировать
```

Главная точка входа для одной карты:

```powershell
python -m services.map_digitizer.pipeline "C:\path\map.jpg" `
  --output-dir "C:\path\result"
```

Алгоритм карт подробно описан в [docs/MAP_PIPELINE.md](docs/MAP_PIPELINE.md).
Архитектура всего сервиса описана в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Где смотреть результат

Последний контрольный результат находится здесь:

```text
results/384092/sheet_21/
  digitized_map.png
  isoline_mask.png
  profile_mask.png
  isolines_pixels.geojson
  surface_local_pixels.cps3
```

`results/` содержит только пользовательские артефакты. Промежуточные JSON и
кеши находятся в `runs/`, потому что API использует их для resume и статусов.

## Правило разработки

Для карт существует один алгоритм и один CLI: `services.map_digitizer.pipeline`.
Новые альтернативные трассировщики не добавляются в production. Эксперимент
сначала живёт вне репозитория; удачный код заменяет соответствующий этап
основного пайплайна.

Полный набор исторических экспериментов и тестовых прогонов вынесен в:

```text
C:\FINAM\GeoArchiveAI_archive\2026-07-17_cleanup
```
