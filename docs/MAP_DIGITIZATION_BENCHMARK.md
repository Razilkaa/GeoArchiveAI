# Map digitization benchmark

Snapshot: 2026-07-17, pipeline v3.

## Dataset

- 30 unique scans from multiple archival reports and map styles.
- 29 current v3 manifests; one v1 manifest is an isolated API smoke job.
- 14 scans produced a surface; 16 were rejected as not applicable.
- Sheets 21 and 23 of report 384092 are included in v3.

## Results

- Statuses: 1 accepted, 13 review, 16 not applicable.
- Different-level contour crossing rate: 0% across all 14 reconstructed surfaces.
- Warm-cache latency: 5.527 s median, 18.943 s P95, 25.960 s maximum.
- A cold FastAPI smoke run including GPU OCR completed in 9.334 s, then exported
  an accepted 118,249-cell EPSG:28421 CPS-3 grid from four control points.
- Updated report batches completed with zero failed pages. Known insufficient-data
  errors are normalized to `not_applicable` rather than aborting a report.

The 3.3% auto-accept rate is intentional. Sparse contour-label reconstruction is
useful for preview and manual/model QC, but it is not auto-approved because one
sheet may contain several geological horizons. Dense reconstruction requires at
least 80 profile measurements and must pass topology, range, interval, conflict
and residual checks.

## Reproduce

```powershell
python -m services.map_digitizer.benchmark runs `
  --output runs/map_validation/final_all_runs_benchmark.json
```

The JSON contains per-sheet reasons, pipeline versions, reconstruction mode,
constraint P95, contour levels, closures, crossings and latency. Generated run
artifacts are intentionally not committed.
