# Map digitizer

Изолированный сервисный модуль для трассировки изолиний, присвоения значений и
экспорта результата map-агента. Сейчас это библиотека с контрактом и тестами;
HTTP-обёртка будет добавлена после стабилизации качества на нескольких картах.

Экспериментальные детекторы и артефакты находятся в
`research/map_digitization/` и не входят в продуктовый pipeline.

## Геопривязка профильной сети

Для контрольного листа 23 эксперимент воспроизводится тремя стадиями:

```powershell
python research/map_digitization/tracing/detect_profile_lines.py <scan> `
  --output pipeline/tracing/out/23/profile_lines.json
python research/map_digitization/tracing/match_profile_labels.py `
  --readings pipeline/output_v4/23/readings.jsonl `
  --lines pipeline/tracing/out/23/profile_lines.json `
  --inventory pipeline/tracing/out/profiles_384092.geojson `
  --output pipeline/tracing/out/23/profile_matches.json
python research/map_digitization/tracing/georeference_profiles_23.py `
  --lines pipeline/tracing/out/23/profile_lines.json `
  --matches pipeline/tracing/out/23/profile_matches.json `
  --inventory pipeline/tracing/out/profiles_384092.geojson `
  --isolines pipeline/tracing/out/23/isolines_23.geojson `
  --gpkg pipeline/tracing/out/23/sheet_23_provisional.gpkg `
  --qc pipeline/tracing/out/23/georeference_qc.json `
  --preview pipeline/tracing/out/23/georeference_preview.png
```

GeoPackage использует `EPSG:2509` и содержит слои изолиний, кандидатов в
сейсмические профили и опорных профилей. Результат остаётся в статусе `review`,
пока третий, геометрически найденный поперечный профиль не подтверждён подписью.
`map_trace_adapter.py` автоматически публикует GeoPackage и превью при наличии
`georeference_qc.json`.
