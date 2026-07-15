"""Reports for detector v4: v3/v4 comparison, 100-box validation montage from three
map areas, suspected false-positive and merged-box lists."""
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")
V3 = Path(r"C:\FINAM\Conference\pipeline\output\21\manifest.jsonl")
OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")

REGION = (2700, 2700, 3900, 3600)  # dense centre used in earlier experiments


def imread_u(path, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)


def imwrite_u(path, img):
    cv2.imencode(Path(path).suffix, img)[1].tofile(str(path))


def side_len(quad):
    d = [math.dist(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    a = (d[0] + d[2]) / 2
    b = (d[1] + d[3]) / 2
    return max(a, b), min(a, b)


img = imread_u(SRC)
man4 = [json.loads(l) for l in (OUT / "manifest_v4.jsonl").read_text(encoding="utf-8").splitlines()]
man3 = [json.loads(l) for l in V3.read_text(encoding="utf-8").splitlines()]
print("v3 boxes:", len(man3), " v4 boxes:", len(man4))

# ---------- comparison image ----------
x0, y0, x1, y1 = REGION
base = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
left, right = base.copy(), base.copy()
for m in man3:
    x, y, w, h = m["bbox"]
    if x + w < x0 or x > x1 or y + h < y0 or y > y1:
        continue
    col = (0, 0, 255) if m["band"] == "small" else (0, 170, 0)
    cv2.rectangle(left, (x - x0, y - y0), (x + w - x0, y + h - y0), col, 2)
for m in man4:
    q = np.array(m["quad"])
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    if not (x0 <= cx <= x1 and y0 <= cy <= y1):
        continue
    col = (0, 0, 255) if m["font_band"] == "small" else \
          (0, 170, 0) if m["font_band"] == "big" else (255, 0, 0)
    cv2.polylines(right, [(q - (x0, y0)).astype(np.int32)], True, col, 2)
for im_, t in ((left, "v3: axis-aligned"), (right, "v4: oriented")):
    cv2.putText(im_, t, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
sep = np.full((left.shape[0], 6, 3), 60, np.uint8)
imwrite_u(OUT / "comparison_v3_v4.png", np.hstack([left, sep, right]))
print("comparison saved")

# ---------- sample 100 boxes from three map areas ----------
random.seed(42)


def centre(m):
    q = np.array(m["quad"])
    return q[:, 0].mean(), q[:, 1].mean()


dense, iso, prof = [], [], []
for m in man4:
    if m["zone"] != "map_body":
        continue
    cx, cy = centre(m)
    if 2400 <= cx <= 4200 and 2400 <= cy <= 3800:
        dense.append(m)
    if m["font_band"] == "big":
        iso.append(m)
    if m["font_band"] == "small" and m["n_components"] >= 5:
        prof.append(m)

sample = []
for name, pool, k in (("dense_center", dense, 34), ("isoline_area", iso, 33),
                      ("profile_numbers", prof, 33)):
    take = random.sample(pool, min(k, len(pool)))
    sample += [(name, m) for m in take]
    print(f"{name}: pool={len(pool)} sampled={len(take)}")

cell_w, cell_h, cols = 300, 130, 5
rows = (len(sample) + cols - 1) // cols
mont = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
d = ImageDraw.Draw(mont)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    font = ImageFont.load_default()
zone_short = {"dense_center": "C", "isoline_area": "I", "profile_numbers": "P"}
for k, (zname, m) in enumerate(sample):
    crop = Image.open(OUT / m["crop"])
    sc = min((cell_w - 78) / crop.width, (cell_h - 8) / crop.height, 2.0)
    crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))))
    cx0, cy0 = (k % cols) * cell_w, (k // cols) * cell_h
    mont.paste(crop, (cx0 + 72, cy0 + 4))
    d.text((cx0 + 4, cy0 + 30), f"#{m['id']}", fill="red", font=font)
    d.text((cx0 + 4, cy0 + 52), f"{zone_short[zname]} n={m['n_components']}", fill="gray", font=font)
    d.text((cx0 + 4, cy0 + 74), f"c={m['detector_confidence']}", fill="gray", font=font)
    d.rectangle([cx0, cy0, cx0 + cell_w - 1, cy0 + cell_h - 1], outline="lightgray")
mont.save(OUT / "validation_montage_100.png")

tmpl = OUT / "val_labels_template.csv"
if tmpl.exists():  # never clobber manual labels: version the new template instead
    k = 2
    while (OUT / f"val_labels_template_v{k}.csv").exists():
        k += 1
    tmpl = OUT / f"val_labels_template_v{k}.csv"
with open(tmpl, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "area", "verdict(ok/fp/merged/split/partial)", "comment"])  # noqa
    for zname, m in sample:
        w.writerow([m["id"], zname, "", ""])
print("validation montage:", len(sample), "boxes")

# ---------- suspects ----------
fp_rows, merged_rows = [], []
for m in man4:
    major, minor = side_len(m["quad"])
    est_chars = major / max(6.0, 0.62 * minor)
    reasons = []
    if m["detector_confidence"] < 0.45:
        reasons.append("low_confidence")
    if m["n_components"] == 2 and m["font_band"] == "unknown":
        reasons.append("two_comps_mixed_band")
    if m["zone"] in ("legend", "title_block", "frame"):
        reasons.append(f"zone_{m['zone']}")
    if reasons:
        fp_rows.append([m["id"], m["detector_confidence"], m["n_components"],
                        m["zone"], ";".join(reasons)])
    if est_chars > 8.5 or m["n_components"] >= 9:
        merged_rows.append([m["id"], round(est_chars, 1), m["n_components"], m["zone"]])

with open(OUT / "suspected_false_positives.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "confidence", "n_components", "zone", "reason"])
    w.writerows(fp_rows)
with open(OUT / "suspected_merged.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "est_chars", "n_components", "zone"])
    w.writerows(merged_rows)

zones = {}
bands = {}
for m in man4:
    zones[m["zone"]] = zones.get(m["zone"], 0) + 1
    bands[m["font_band"]] = bands.get(m["font_band"], 0) + 1
print("zones:", zones)
print("bands:", bands)
print("suspected FP:", len(fp_rows), " suspected merged:", len(merged_rows))
