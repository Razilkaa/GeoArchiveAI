"""The geophysically honest product: depth values ON PROFILES.
Intersect valued isolines with digitized profile lines -> points
(x, y, depth). The map is then OUR interpolation of these points.
Compare our surface with the author's scan."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point

Image.MAX_IMAGE_PIXELS = None
OUT = Path(r"C:\FINAM\Conference\pipeline\tracing\out")
TR = OUT / "23"
BMP = Path(r"C:\FINAM\Conference\384092\Графика\Том 2, 3 (выборка)\23.bmp")

iso_all = json.loads((TR / "isolines.json").read_text(encoding="utf-8"))
clean = json.loads((TR / "isolines_clean.json").read_text(encoding="utf-8"))
va = json.loads((TR / "valued_isolines_23.json").read_text(encoding="utf-8"))
cc = json.loads((TR / "crosscheck_23.json").read_text(encoding="utf-8"))

# valued isolines (label + surface-propagated), cleaned indexing
polys_clean = [np.array(p, float) for p in clean["polylines_xy"]]
values = {int(k): abs(v) for k, v in va["values"].items()}
values.update({int(k): abs(v) for k, v in cc["from_surface"].items()})
verd = {int(k): v for k, v in cc["verdicts"].items()}
iso_lines = {i: LineString(p) for i, p in enumerate(polys_clean)
             if i in values and verd.get(i) != "flagged"}
print("valued isolines used:", len(iso_lines))

# profiles = curves removed by crossing filter, EXTENDED as fitted straight
# lines across the sheet (profiles are straight; digitized fragments are short)
polys_orig = [np.array(p, float) for p in iso_all["polylines_xy"]]
img_probe = np.array(Image.open(BMP).convert("L"))
Hs, Ws = img_probe.shape
DIAG = float(np.hypot(Ws, Hs))
profiles = []
for r in clean["removed"]:
    if r["straight_likeness"] <= 0.5:
        continue
    p = polys_orig[r["id"]]
    mean = p.mean(0)
    _, _, vt = np.linalg.svd(p - mean)
    u = vt[0]
    a = mean - u * DIAG
    b = mean + u * DIAG
    profiles.append(LineString([tuple(a), tuple(b)]))
print("digitized profiles (extended):", len(profiles))

# intersections
pts = []
for pid, prof in enumerate(profiles):
    for i, line in iso_lines.items():
        inter = prof.intersection(line)
        geoms = []
        if isinstance(inter, Point):
            geoms = [inter]
        elif isinstance(inter, MultiPoint):
            geoms = list(inter.geoms)
        for g in geoms:
            pts.append((g.x, g.y, values[i], pid, verd.get(i, "")))
print("profile x isoline points:", len(pts))

with open(TR / "profile_points_23.csv", "w", encoding="utf-8-sig") as f:
    f.write("x_px,y_px,depth_km,profile_id,verdict\n")
    for x, y, v, pid, vd in pts:
        f.write(f"{x:.1f},{y:.1f},{-v},{pid},{vd}\n")

# OUR interpolation: linear inside the data hull (no extrapolation at all)
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

arr = np.array([(x, y, v) for x, y, v, _, _ in pts])
img = np.array(Image.open(BMP).convert("L"))
H, W = img.shape
gx, gy = np.meshgrid(np.linspace(0, W, 550), np.linspace(0, H, 540))
gz = griddata(arr[:, :2], arr[:, 2], (gx, gy), method="linear")
gz = gaussian_filter(np.nan_to_num(gz, nan=0.0), 1.6) + (gz * 0)  # smooth, keep NaN mask
d, _ = cKDTree(arr[:, :2]).query(np.column_stack([gx.ravel(), gy.ravel()]))
gz[(d > 500).reshape(gx.shape)] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=110)
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Оригинал (лист 23, Шеинская)")
axes[0].axis("off")
ax = axes[1]
ax.imshow(img, cmap="gray", alpha=0.22)
levels = np.arange(np.floor(np.nanmin(gz) * 5) / 5, np.nanmax(gz) + 0.1, 0.2)
cf = ax.contourf(gx, gy, gz, levels=levels, cmap="turbo_r", alpha=0.7)
cs = ax.contour(gx, gy, gz, levels=levels, colors="k", linewidths=0.7)
ax.clabel(cs, fmt="-%.1f", fontsize=7)
ax.scatter(arr[:, 0], arr[:, 1], s=6, c="k", alpha=0.8)
ax.set_title(f"НАША интерполяция по {len(pts)} точкам «профиль x изогипса»\n"
             "(значения существуют только на профилях — карта строится заново)")
ax.axis("off")
fig.colorbar(cf, ax=ax, shrink=0.6, label="Глубина К, км (абс.)")
fig.tight_layout()
fig.savefig(TR / "profile_points_surface_23.png", bbox_inches="tight", facecolor="white")
print("saved:", TR / "profile_points_surface_23.png")
