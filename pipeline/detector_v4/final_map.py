"""Honest presentation map: clean extracted data only, drawn in classic
structural-map style (white background, green isolines, blue profiles)."""
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


def take(kind, band=None):
    out = []
    for r in rows:
        if r["kind"] != kind or r["zone"] != "map_body" or r["qc_flag"] or not r["x"]:
            continue
        if band and r["font_band"] != band:
            continue
        out.append((float(r["x"]), float(r["y"]), abs(float(r["value"]))))
    return np.array(out)


marks = take("depth_mark")
iso = take("isoline", band="big")
pts = np.vstack([marks, iso])
wells = [(float(r["x"]), float(r["y"]), r["value"]) for r in rows
         if r["kind"] == "well" and r["x"]]
print(f"depth marks: {len(marks)}, isoline labels: {len(iso)}, wells: {len(wells)}")

img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape

# --- straight profile lines from the raster (long-component mask + Hough) ---
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
long_mask = np.zeros_like(bw)
for i in range(1, n):
    x, y, w, h, a = stats[i]
    if max(w, h) >= 400:
        long_mask[lab == i] = 255
segs = cv2.HoughLinesP(long_mask, 1, np.pi / 360, threshold=220,
                       minLineLength=600, maxLineGap=25)
segs = segs.reshape(-1, 4) if segs is not None else []
print("profile segments:", len(segs))

# --- surface ---
rbf = RBFInterpolator(pts[:, :2], pts[:, 2], kernel="thin_plate_spline",
                      smoothing=2.0, neighbors=60)
gx, gy = np.meshgrid(np.linspace(0, W, 800), np.linspace(0, H, 690))
gz = rbf(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
gz = gaussian_filter(gz, 1.2)
d, _ = cKDTree(pts[:, :2]).query(np.column_stack([gx.ravel(), gy.ravel()]))
gz[(d > 300).reshape(gx.shape)] = np.nan

# --- draw ---
fig, ax = plt.subplots(figsize=(16, 13), dpi=140)
ax.set_facecolor("white")
S = 6.0

for x1, y1, x2, y2 in segs:
    ax.plot([x1 / S, x2 / S], [y1 / S, y2 / S], color="#4169b0", lw=0.7, alpha=0.8, zorder=1)

levels = np.arange(np.floor(np.nanmin(gz) * 5) / 5, np.nanmax(gz) + 0.1, 0.1)
cs_min = ax.contour(gx / S, gy / S, gz, levels=levels, colors="#1a7a2e",
                    linewidths=0.5, alpha=0.75, zorder=2)
cs_maj = ax.contour(gx / S, gy / S, gz, levels=levels[::2], colors="#146423",
                    linewidths=1.1, zorder=3)
ax.clabel(cs_maj, fmt="-%.1f", fontsize=8, colors="#146423")

for x, y, name in wells:
    ax.scatter(x / S, y / S, s=60, facecolors="none", edgecolors="#c81abe",
               linewidths=1.6, zorder=5)
    ax.annotate(name, (x / S, y / S), xytext=(6, 6), textcoords="offset points",
                fontsize=8, color="#c81abe", weight="bold")

ax.set_xlim(0, W / S)
ax.set_ylim(H / S, 0)
ax.set_aspect("equal")
ax.axis("off")

# title block (honest)
tb = (
    "СТРУКТУРНАЯ КАРТА\n"
    "по отражающему горизонту К\n"
    "Хампинская площадь\n"
    "Источник: прил. 21, Хампинская с/п 15/79-80, 1980 г.\n"
    "Автоматическая оцифровка (CV + VLM), 2026 г.\n"
    f"{len(marks)} отметок глубин, {len(iso)} подписей изогипс\n"
    "Сечение изогипс 0.2 км. Координаты: пиксели скана\n"
    "БЕЗ ручной корректуры. QC: медианное расх. с подписями 0.2 км"
)
ax.text(0.985, 0.02, tb, transform=ax.transAxes, fontsize=9, family="serif",
        ha="right", va="bottom",
        bbox=dict(boxstyle="square,pad=0.6", fc="white", ec="black", lw=1.2))
leg = (
    "Условные обозначения:\n"
    "──── изогипсы горизонта К (реконструкция), км\n"
    "──── сейсмопрофили (детекция прямых на растре)\n"
    "  ○   скважины (прочитаны со скана)\n"
    "  ·    точки автоматически извлечённых отметок"
)
ax.text(0.015, 0.02, leg, transform=ax.transAxes, fontsize=9, family="serif",
        ha="left", va="bottom",
        bbox=dict(boxstyle="square,pad=0.6", fc="white", ec="black", lw=1.0))
ax.scatter(pts[:, 0] / S, pts[:, 1] / S, s=0.8, c="#666666", alpha=0.35, zorder=2)

fig.tight_layout()
fig.savefig(OUT / "final_map_21.png", bbox_inches="tight", facecolor="white")
print("saved:", OUT / "final_map_21.png")
