"""Surface v2: outlier rejection by local median, smoothing RBF instead of
linear triangulation, blanking mask over data gaps."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")

BLANK_DIST = 260        # px: no drawing farther than this from any data point
OUTLIER_K = 8           # neighbours for local median
OUTLIER_TOL = 0.22      # km: reject if |value - local median| exceeds this

rows = list(csv.DictReader(open(OUT / "points_final.csv", encoding="utf-8-sig")))
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
print("input points:", len(pts))

# local-median outlier rejection
tree = cKDTree(pts[:, :2])
_, idx = tree.query(pts[:, :2], k=OUTLIER_K + 1)
med = np.median(pts[idx[:, 1:], 2], axis=1)
keep = np.abs(pts[:, 2] - med) <= OUTLIER_TOL
print(f"outliers rejected: {int((~keep).sum())} ({(~keep).mean():.1%})")
pts = pts[keep]

img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape

# smoothing RBF on a subsample-free full set
rbf = RBFInterpolator(pts[:, :2], pts[:, 2], kernel="thin_plate_spline",
                      smoothing=2.0, neighbors=60)
gx, gy = np.meshgrid(np.linspace(0, W, 700), np.linspace(0, H, 600))
gz = rbf(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
gz = gaussian_filter(gz, 1.2)

# blanking: hide cells far from data
gtree = cKDTree(pts[:, :2])
dist, _ = gtree.query(np.column_stack([gx.ravel(), gy.ravel()]))
gz[(dist > BLANK_DIST).reshape(gx.shape)] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=150)
axes[0].imshow(cv2.resize(img, (W // 6, H // 6)), cmap="gray")
axes[0].set_title("Оригинал: скан 1980 г.", fontsize=13)
axes[0].axis("off")

ax = axes[1]
ax.imshow(cv2.resize(img, (W // 6, H // 6)), cmap="gray", alpha=0.25)
levels = np.arange(np.floor(np.nanmin(gz) * 10) / 10, np.nanmax(gz) + 0.1, 0.1)
cf = ax.contourf(gx / 6, gy / 6, gz, levels=levels, cmap="RdYlBu", alpha=0.75)
cs = ax.contour(gx / 6, gy / 6, gz, levels=levels[::2], colors="k", linewidths=0.6)
ax.clabel(cs, fmt="%.1f", fontsize=6)
ax.scatter(pts[:, 0] / 6, pts[:, 1] / 6, s=1.0, c="k", alpha=0.45)
ax.set_title(f"Реконструкция v2: {len(pts)} точек, фильтр выбросов, RBF, blanking",
             fontsize=13)
ax.axis("off")
fig.colorbar(cf, ax=ax, shrink=0.7, label="Глубина горизонта К, км")
fig.tight_layout()
fig.savefig(OUT / "digitized_map_21_v2.png", bbox_inches="tight", facecolor="white")
print("saved:", OUT / "digitized_map_21_v2.png")
