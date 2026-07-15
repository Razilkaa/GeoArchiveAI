# Report Factory

The production path is optimized for time-to-first-result, not full archival OCR.

1. Inventory every scan and create a resumable `runs/<report_id>/job.json`.
2. Route graphics from low-resolution previews in VLM batches.
3. OCR explicit contents pages, front matter, conclusions, and appendices first.
4. Use the contents and first retrieval results to add only relevant text pages.
5. Ingest provenance-preserving Markdown into RAGFlow.
6. Run structure, well, map, horizon, and key-result agents in parallel.
7. Return one result bundle; full OCR and map digitization are optional background jobs.

New reports go into `C:\FINAM\Conference\reports_inbox`, one directory per report.

```powershell
python pipeline\report_factory.py reports_inbox\REPORT_ID
python pipeline\page_router.py runs\REPORT_ID\job.json --credentials-file secrets\tokent.txt
```

The resumable worker runs the complete core path and skips valid cached stages:

```powershell
python pipeline\job_worker.py REPORT_ID --allow-external
python pipeline\job_worker.py --all --allow-external
```

Without `--allow-external`, report privacy policy must explicitly allow external models.
Map tracing and full archival OCR remain optional background branches.

The inbox watcher starts this worker automatically. Reports without an upload policy
remain blocked before any external VLM call. For a trusted archive copied manually:

```powershell
pipeline\copy_to_inbox.ps1 -SourceFile D:\reports\fund.rar -AllowExternalModels
```
