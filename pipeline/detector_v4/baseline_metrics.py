"""Baseline metrics of the naive per-crop VLM reading run (for the theses).

Latency per request is present only in records written after read_crops.py was
instrumented; for older records the average is derived from wall-clock duration.
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
WALL_START = sys.argv[1] if len(sys.argv) > 1 else None  # "HH:MM" launch time, optional

RE_PICKET = re.compile(r"^-?[0-9]\.[0-9]{2}$")
RE_ISOLINE = re.compile(r"^-?[0-9]\.[0-9]$")
RE_PROFILE = re.compile(r"^[0-9]{5,6}$")
RE_WELL = re.compile(r"^[РP]-?[0-9]{2,4}$", re.I)


def classify(text):
    t = str(text).strip().replace(",", ".").replace("−", "-")
    if RE_PICKET.match(t):
        return "picket"
    if RE_ISOLINE.match(t):
        return "isoline"
    if RE_PROFILE.match(t):
        return "profile_id"
    if RE_WELL.match(t):
        return "well"
    return "other"


rows = [json.loads(l) for l in (OUT / "readings.jsonl").read_text(encoding="utf-8").splitlines()]
n = len(rows)
errors = sum(1 for r in rows if r.get("error"))
unreadable = sum(1 for r in rows if not r.get("error") and (r.get("unreadable") or not r.get("values")))
with_values = n - errors - unreadable
values = [v for r in rows if r.get("values") for v in r["values"]]
kinds = Counter(classify(v.get("text", "")) for v in values)
lat = [r["latency_s"] for r in rows if r.get("latency_s")]

mtime = datetime.fromtimestamp((OUT / "readings.jsonl").stat().st_mtime)
metrics = {
    "run": "naive per-crop VLM baseline",
    "model": rows[-1].get("model", "gpt-4o-mini"),
    "sheet": "384092 / Том 2 / прил. 21 (структурная карта, 8657x7498 px)",
    "requests": n,
    "boxes_with_values": with_values,
    "boxes_unreadable": unreadable,
    "boxes_error": errors,
    "values_total": len(values),
    "values_by_kind": dict(kinds.most_common()),
    "latency_recorded_requests": len(lat),
    "latency_avg_s": round(sum(lat) / len(lat), 2) if lat else None,
    "finished_at": mtime.isoformat(timespec="seconds"),
    "wall_start_hint": WALL_START,
}
(OUT / "baseline_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
