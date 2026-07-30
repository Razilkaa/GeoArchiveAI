# RAG quality regression

The evaluator calls the public FastAPI contract. Retrieval stays in RAGFlow;
there is no local BM25 or secondary index.

Build a review draft from evidence-backed entities:

```powershell
python evaluation/rag/build_gold_candidates.py --limit 100
```

An expert must verify the question, expected pages and required terms, then set
`review_status` to `approved`. Candidate-only diagnostic runs are explicit:

```powershell
python evaluation/rag/run_regression.py --include-candidates --limit 10
```

Run the approved retrieval baseline:

```powershell
python evaluation/rag/run_regression.py
```

Release baselines are sequential by default. In the current environment,
`--workers 2` already exposes upstream capacity failures; use higher concurrency
only as an explicit load test.

Sequential runs checkpoint `report.json` after every case. A stopped run can be
continued with the same `--output-dir ... --resume` arguments.

Use `--mode live` to additionally score required terms, forbidden terms,
abstention and citation QC. Reports are published under `results/rag_evaluation/`.

The first diagnostic result is summarized in `BASELINE.md`. Release gating should
use both the successful-call metrics and the effective metrics that count upstream
errors as misses.
