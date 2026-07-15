"""Merge pass-1 readings with the context re-read and bake ALL QC flags into
points_final.csv (single source of truth for downstream scripts).

Classes: depth_mark (отметка глубины на пикете, формат d.dd), isoline (подпись
изогипсы d.d), profile_id (5-6 цифр), well (Р-410), station (порядковый номер
пикета/станции: целое 1-3 цифры или ПК-формат), other.
QC flags (через ;): band_mismatch, out_of_range, local_outlier, spatial_dup.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")

RE_DEPTH = re.compile(r"^-?[0-9]\.[0-9]{2}$")
RE_ISOLINE = re.compile(r"^-?[0-9]\.[0-9]$")
RE_PROFILE = re.compile(r"^[0-9]{5,6}$")
RE_WELL = re.compile(r"^[РP]-?[0-9]{2,4}$", re.I)
RE_STATION = re.compile(r"^([0-9]{1,3}|[ПP]К ?[0-9/]+)$", re.I)

DEPTH_RANGE = (1.8, 4.8)   # km, plausible for horizon K on this sheet
OUTLIER_K = 8
OUTLIER_TOL = 0.22
DUP_CELL = 120


def classify(text):
    t = str(text).strip().replace(",", ".").replace("−", "-")
    if RE_DEPTH.match(t):
        return "depth_mark", t
    if RE_ISOLINE.match(t):
        return "isoline", t
    if RE_PROFILE.match(t):
        return "profile_id", t
    if RE_WELL.match(t):
        return "well", t
    if RE_STATION.match(t):
        return "station", t
    return "other", t


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


p1 = {r["id"]: r for r in load(OUT / "readings.jsonl")}
p2 = {r["id"]: r for r in load(OUT / "readings_reread.jsonl")}
merged = {**p1, **p2}

pts = []
unreadable = 0
for r in merged.values():
    if r.get("unreadable") or not r.get("values"):
        unreadable += 1
        continue
    q = r.get("quad")
    cx = sum(p[0] for p in q) / 4 if q else None
    cy = sum(p[1] for p in q) / 4 if q else None
    for v in r["values"]:
        kind, norm = classify(v.get("text", ""))
        flags = []
        if kind == "isoline" and r.get("font_band") == "small":
            flags.append("band_mismatch")
        if kind == "depth_mark" and r.get("font_band") == "big":
            flags.append("band_mismatch")
        if kind == "depth_mark" and not (DEPTH_RANGE[0] <= abs(float(norm)) <= DEPTH_RANGE[1]):
            flags.append("out_of_range")
        pts.append({
            "box_id": r["id"], "value": norm, "kind": kind, "flags": flags,
            "vlm_conf": v.get("confidence"),
            "x": round(cx, 1) if cx else None, "y": round(cy, 1) if cy else None,
            "zone": r.get("zone"), "font_band": r.get("font_band"),
            "pass": r.get("pass", "pass1"),
        })

# spatial duplicate flag (same value nearby - context-crop overlap)
seen = set()
for p in pts:
    if p["x"] is None:
        continue
    key = (p["kind"], p["value"], round(p["x"] / DUP_CELL), round(p["y"] / DUP_CELL))
    if key in seen:
        p["flags"].append("spatial_dup")
    else:
        seen.add(key)

# local-outlier flag for depth marks (on non-dup, in-range, map_body points)
dm = [p for p in pts if p["kind"] == "depth_mark" and p["zone"] == "map_body"
      and p["x"] is not None and "spatial_dup" not in p["flags"]
      and "out_of_range" not in p["flags"]]
xy = np.array([(p["x"], p["y"]) for p in dm])
vals = np.array([abs(float(p["value"])) for p in dm])
tree = cKDTree(xy)
_, idx = tree.query(xy, k=OUTLIER_K + 1)
med = np.median(vals[idx[:, 1:]], axis=1)
for p, v, m in zip(dm, vals, med):
    if abs(v - m) > OUTLIER_TOL:
        p["flags"].append("local_outlier")

for p in pts:
    p["qc_flag"] = ";".join(p["flags"])
    del p["flags"]

with open(OUT / "points_final.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(pts[0].keys()))
    w.writeheader()
    w.writerows(pts)

body = [p for p in pts if p["zone"] == "map_body"]
print("values total:", len(pts), " map_body:", len(body), " unreadable boxes:", unreadable)
print("by kind:", dict(Counter(p["kind"] for p in body).most_common()))
clean = [p for p in body if not p["qc_flag"]]
print("clean (no flags):", dict(Counter(p["kind"] for p in clean).most_common()))
print("flag counts:", dict(Counter(f for p in body for f in p["qc_flag"].split(";") if f)))
