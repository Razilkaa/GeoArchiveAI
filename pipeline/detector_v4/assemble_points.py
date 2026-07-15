"""Assemble read values into classified points.

readings.jsonl (VLM output) -> points_v4.csv + stats + value overlay.
Classification is re-derived from value format (regex); the VLM's own 'kind'
is kept for comparison but format wins.
"""
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")

RE_PICKET = re.compile(r"^-?[0-9]\.[0-9]{2}$")       # 2.72, 3.08
RE_ISOLINE = re.compile(r"^-?[0-9]\.[0-9]$")          # -2.8, 4.0
RE_PROFILE = re.compile(r"^[0-9]{5,6}$")              # 40412, 311182
RE_WELL = re.compile(r"^[РP]-?[0-9]{2,4}$", re.I)     # Р-410 (cyr/lat P)
RE_PK = re.compile(r"^[ПP]К", re.I)                   # ПК... пикетаж


def classify(text):
    t = text.strip().replace(",", ".").replace("−", "-")
    if RE_PICKET.match(t):
        return "picket", t
    if RE_ISOLINE.match(t):
        return "isoline", t
    if RE_PROFILE.match(t):
        return "profile_id", t
    if RE_WELL.match(t):
        return "well", t
    if RE_PK.match(t):
        return "pk_mark", t
    return "other", t


def main():
    rows = [json.loads(l) for l in (OUT / "readings.jsonl").read_text(encoding="utf-8").splitlines()]
    print("readings:", len(rows))

    pts = []
    stats = {"boxes_read": 0, "boxes_unreadable": 0, "boxes_error": 0}
    kinds = {}
    agree = disagree = 0
    for r in rows:
        if r.get("error"):
            stats["boxes_error"] += 1
            continue
        if r.get("unreadable") or not r.get("values"):
            stats["boxes_unreadable"] += 1
            continue
        stats["boxes_read"] += 1
        q = np.array(r["quad"]) if "quad" in r else None
        cx, cy = (q[:, 0].mean(), q[:, 1].mean()) if q is not None else (None, None)
        for v in r["values"]:
            kind, norm = classify(str(v.get("text", "")))
            kinds[kind] = kinds.get(kind, 0) + 1
            if v.get("kind") == kind:
                agree += 1
            else:
                disagree += 1
            pts.append({
                "box_id": r["id"], "value": norm, "kind": kind,
                "vlm_kind": v.get("kind"), "vlm_conf": v.get("confidence"),
                "x": round(float(cx), 1) if cx is not None else "",
                "y": round(float(cy), 1) if cy is not None else "",
                "angle": r.get("angle"), "zone": r.get("zone"),
                "font_band": r.get("font_band"),
                "det_conf": r.get("detector_confidence"),
                "n_values_in_box": len(r["values"]),
            })

    with open(OUT / "points_v4.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0].keys()))
        w.writeheader()
        w.writerows(pts)

    print("boxes:", stats)
    print("values by kind:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    print(f"kind agreement vlm vs regex: {agree}/{agree + disagree}")

    # overlay of classified values on downscaled sheet
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    scale = 6
    ov = cv2.cvtColor(cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale)),
                      cv2.COLOR_GRAY2BGR)
    col = {"picket": (0, 0, 255), "isoline": (0, 170, 0), "profile_id": (255, 0, 0),
           "well": (200, 0, 200), "pk_mark": (0, 140, 255), "other": (160, 160, 160)}
    for p in pts:
        if p["x"] == "" or p["zone"] != "map_body":
            continue
        c = col[p["kind"]]
        cv2.circle(ov, (int(p["x"] / scale), int(p["y"] / scale)), 3, c, -1)
    cv2.imencode(".png", ov)[1].tofile(str(OUT / "points_overlay_v4.png"))
    print("overlay:", OUT / "points_overlay_v4.png")


if __name__ == "__main__":
    main()
