"""Quick maps from control points: isoline labels only / depth marks only / combined."""
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

rows = list(csv.DictReader(open(OUT / "points_final.csv", encoding="utf-8-sig")))


def collect(kind, band=None, rng=(1.4, 5.0), cell=120):
    pts, seen = [], set()
    for r in rows:
        if r["kind"] != kind or r["zone"] != "map_body" or r["qc_flag"] or not r["x"]:
            continue
        if band and r["font_band"] != band:
            continue
        v = abs(float(r["value"]))
        if not (rng[0] <= v <= rng[1]):
            continue
        key = (r["value"], round(float(r["x"]) / cell), round(float(r["y"]) / cell))
        if key in seen:
            continue
        seen.add(key)
        pts.append((float(r["x"]), float(r["y"]), v))
    return np.array(pts)


marks = collect("picket", rng=(1.8, 4.8))
tree = cKDTree(marks[:, :2])
_, idx = tree.query(marks[:, :2], k=9)
med = np.median(marks[idx[:, 1:], 2], axis=1)
marks = marks[np.abs(marks[:, 2] - med) <= 0.22]
iso = collect("isoline", band="big", cell=150)
both = np.vstack([marks, iso])
print(f"depth marks: {len(marks)}, isoline labels: {len(iso)}")

img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape
gx, gy = np.meshgrid(np.linspace(0, W, 600), np.linspace(0, H, 520))
grid_xy = np.column_stack([gx.ravel(), gy.ravel()])


def surface(pts, smooth, blank):
    rbf = RBFInterpolator(pts[:, :2], pts[:, 2], kernel="thin_plate_spline",
                          smoothing=smooth, neighbors=min(60, len(pts) - 1))
    gz = rbf(grid_xy).reshape(gx.shape)
    gz = gaussian_filter(gz, 1.0)
    d, _ = cKDTree(pts[:, :2]).query(grid_xy)
    gz[(d > blank).reshape(gx.shape)] = np.nan
    return gz


panels = [
    (f"По {len(iso)} подписям изогипс (опорные)", surface(iso, 0.5, 900), iso),
    (f"По {len(marks)} отметкам глубин на пикетах", surface(marks, 2.0, 260), marks),
    (f"Совмещённо: {len(both)} точек", surface(both, 2.0, 300), both),
]

fig, axes = plt.subplots(1, 3, figsize=(27, 8.5), dpi=120)
levels = np.arange(1.9, 4.8, 0.1)
for ax, (title, gz, pts) in zip(axes, panels):
    ax.imshow(cv2.resize(img, (W // 6, H // 6)), cmap="gray", alpha=0.25)
    cf = ax.contourf(gx / 6, gy / 6, gz, levels=levels, cmap="RdYlBu", alpha=0.75)
    cs = ax.contour(gx / 6, gy / 6, gz, levels=levels[::2], colors="k", linewidths=0.5)
    ax.clabel(cs, fmt="%.1f", fontsize=6)
    ax.scatter(pts[:, 0] / 6, pts[:, 1] / 6, s=2.5, c="k", alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
fig.colorbar(cf, ax=axes, shrink=0.7, label="Глубина, км")
fig.savefig(OUT / "quick_maps_3panel.png", bbox_inches="tight", facecolor="white")
print("saved:", OUT / "quick_maps_3panel.png")
