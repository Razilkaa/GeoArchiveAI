# Оцифровка карт

Запускаемый файл здесь один: `pipeline.py`.

```powershell
python -m services.map_digitizer.pipeline "map.jpg" --output-dir "result"
```

Остальные `.py` не являются альтернативными скриптами. Это внутренние этапы:

| Файл | Роль |
|---|---|
| `ocr_client.py` | один запрос к PaddleOCR |
| `trace_map_isolines.py` | разделение текста, профилей и изолиний |
| `assign_contour_values.py` | привязка OCR-подписей к найденным линиям |
| `depth_mark_surface.py` | опорная поверхность по числам на профилях |
| `trace_guided_surface.py` | интерполяция между исходными трассами |
| `contour_cleanup.py` | удаление неподдержанных замкнутых артефактов |
| `local_exports.py` | итоговая PNG, GeoJSON, XYZ и CPS-3 |
| `georeference_grid.py` | перевод пиксельного Grid в заданную CRS |
| `report_batch.py` | вызов того же pipeline для карт полного отчёта |
| `export_cps3_grid.py` | техническая запись CPS-3 |

Подробное объяснение: `docs/MAP_PIPELINE.md`.
