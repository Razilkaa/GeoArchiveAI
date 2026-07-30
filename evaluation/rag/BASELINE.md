# RAGFlow baseline — 2026-07-18

## Full candidate retrieval run

The first complete run covered 86 evidence-backed candidates across 15 reports
with `workers=2`:

| Metric | Value |
|---|---:|
| Cases | 86 |
| Successful upstream calls | 61 |
| Upstream errors | 25 |
| Hit@5 among successful calls | 0.6885 |
| MRR among successful calls | 0.5574 |
| Effective Hit@5 including errors | 0.4884 |
| Effective MRR including errors | 0.3953 |
| Mean latency of successful calls | 3725 ms |

All 25 errors were FastAPI 502 responses after an upstream retrieval timeout.
Repeating an error case alone succeeded in about 3 seconds, proving that the
current RAGFlow endpoint cannot reliably sustain even two concurrent retrievals.
Release baselines therefore default to `workers=1`; `workers>1` is a load test.

The candidate expected-page lists are intentionally conservative. An entity can
occur on other relevant pages, so these values are a lower bound until an expert
reviews and approves each case.

## Reliability follow-up

Sequential checkpoint/resume was verified on a four-case run. Three cases
completed in about 3 seconds each; one transient request still timed out. The
production client now retries timeout/429/502/503/504 responses with configurable
attempts, timeout and backoff. The evaluator writes an atomic checkpoint after
every sequential case and reports mean, p95 and p99 latency.

Machine-readable outputs are published under `results/rag_evaluation/` and are
intentionally excluded from Git.
