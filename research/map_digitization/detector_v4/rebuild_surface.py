"""The payoff: rebuild the structural surface from extracted picket points
and render it next to the original scan."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")

rows = list(csv.DictReader(open(OUT / "points_final.csv", encoding="utf-8-sig")))

# clean pickets: map body, no qc flag, plausible depth range for horizon K here
pts = []
seen = set()
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
print("clean picket points:", len(pts))

img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape

# grid + linear interpolation inside data cloud
gx, gy = np.meshgrid(np.linspace(0, W, 500), np.linspace(0, H, 430))
gz = griddata(pts[:, :2], pts[:, 2], (gx, gy), method="linear")

fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=150)
axes[0].imshow(cv2.resize(img, (W // 6, H // 6)), cmap="gray")
axes[0].set_title("Оригинал: скан 1980 г.", fontsize=13)
axes[0].axis("off")

ax = axes[1]
ax.imshow(cv2.resize(img, (W // 6, H // 6)), cmap="gray", alpha=0.25)
cf = ax.contourf(gx / 6, gy / 6, gz, levels=np.arange(1.9, 4.7, 0.1),
                 cmap="RdYlBu", alpha=0.75)
cs = ax.contour(gx / 6, gy / 6, gz, levels=np.arange(1.9, 4.7, 0.2),
                colors="k", linewidths=0.6)
ax.clabel(cs, fmt="%.1f", fontsize=6)
ax.scatter(pts[:, 0] / 6, pts[:, 1] / 6, s=1.2, c="k", alpha=0.5)
ax.set_title(f"Реконструкция по {len(pts)} автоматически извлечённым точкам", fontsize=13)
ax.axis("off")
fig.colorbar(cf, ax=ax, shrink=0.7, label="Глубина горизонта К, км")
fig.tight_layout()
fig.savefig(OUT / "digitized_map_21.png", bbox_inches="tight", facecolor="white")
print("saved:", OUT / "digitized_map_21.png")
