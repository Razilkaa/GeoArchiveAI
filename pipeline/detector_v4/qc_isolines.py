"""QC: compare reconstructed surface with trusted isoline labels (big font).
Median mismatch <= half contour interval => the surface can be trusted."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")

rows = list(csv.DictReader(open(OUT / "points_final.csv", encoding="utf-8-sig")))

# same cleaning as rebuild_surface2
pts, seen = [], set()
for r in rows:
    if r["kind"] != "picket" or r["zone"] != "map_body" or r["qc_flag"] or not r["x"]:
        continue
    v = abs(float(r["value"]))
    if not (1.8 <= v <= 4.8):
        continue
    key = (r["value"], round(float(r["x"]) / 120), round(float(r["y"]) / 120))
    if key in seen:
        continue
    seen.add(key)
    pts.append((float(r["x"]), float(r["y"]), v))
pts = np.array(pts)
tree = cKDTree(pts[:, :2])
_, idx = tree.query(pts[:, :2], k=9)
med = np.median(pts[idx[:, 1:], 2], axis=1)
pts = pts[np.abs(pts[:, 2] - med) <= 0.22]
rbf = RBFInterpolator(pts[:, :2], pts[:, 2], kernel="thin_plate_spline",
                      smoothing=2.0, neighbors=60)

# trusted isoline labels
iso = []
seen = set()
for r in rows:
    if r["kind"] != "isoline" or r["zone"] != "map_body" or r["qc_flag"] or not r["x"]:
        continue
    if r["font_band"] != "big":
        continue
    v = abs(float(r["value"]))
    if not (1.4 <= v <= 5.0):
        continue
    key = (r["value"], round(float(r["x"]) / 150), round(float(r["y"]) / 150))
    if key in seen:
        continue
    seen.add(key)
    iso.append((float(r["x"]), float(r["y"]), v))
iso = np.array(iso)
print("trusted isoline labels:", len(iso))

# only labels within data coverage
dist, _ = tree.query(iso[:, :2])
inside = dist < 300
iso_in = iso[inside]
pred = rbf(iso_in[:, :2])
diff = pred - iso_in[:, 2]
print(f"labels inside coverage: {len(iso_in)}")
print(f"mismatch |surface - label|: median={np.median(np.abs(diff)):.3f} km, "
      f"p75={np.percentile(np.abs(diff), 75):.3f}, p90={np.percentile(np.abs(diff), 90):.3f}")
print(f"within 0.1 km (contour interval): {(np.abs(diff) <= 0.10).mean():.0%}")
print(f"within 0.2 km: {(np.abs(diff) <= 0.20).mean():.0%}")

img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape
fig, ax = plt.subplots(figsize=(12, 10), dpi=130)
ax.imshow(cv2.resize(img, (W // 6, H // 6)), cmap="gray", alpha=0.45)
ok = np.abs(diff) <= 0.10
mid = (np.abs(diff) > 0.10) & (np.abs(diff) <= 0.25)
bad = np.abs(diff) > 0.25
for mask, c, lbl in ((ok, "green", "|Δ|≤0.1"), (mid, "orange", "0.1<|Δ|≤0.25"),
                     (bad, "red", "|Δ|>0.25")):
    ax.scatter(iso_in[mask, 0] / 6, iso_in[mask, 1] / 6, s=26, c=c, label=lbl,
               edgecolors="k", linewidths=0.4)
for x, y, v in iso[~inside]:
    ax.scatter(x / 6, y / 6, s=20, c="gray", marker="x")
ax.legend(loc="lower left")
ax.set_title("QC: расхождение реконструкции с подписями изогипс, км")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "qc_isolines.png", bbox_inches="tight", facecolor="white")

with open(OUT / "qc_isolines.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "label_value", "surface_value", "diff"])
    for (x, y, v), p in zip(iso_in, pred):
        w.writerow([int(x), int(y), v, round(float(p), 3), round(float(p - v), 3)])
print("saved qc_isolines.png / .csv")
