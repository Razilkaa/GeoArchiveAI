"""Side-by-side: survey profiles from the shapefile (real coords, СК-42) vs our
digitized straight lines on sheet 23 (pixels). Eyeball geometry before matching."""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(r"C:\FINAM\Conference\pipeline\tracing\out")
gdf = gpd.read_file(OUT / "profiles_384092.geojson")

iso = json.loads((OUT / "23" / "isolines.json").read_text(encoding="utf-8"))
clean = json.loads((OUT / "23" / "isolines_clean.json").read_text(encoding="utf-8"))
removed_ids = [r["id"] for r in clean["removed"]]
polys = [np.array(pl, float) for pl in iso["polylines_xy"]]

fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=110)
ax = axes[0]
for _, row in gdf.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, lw=1.2)
    label = str(row.get("N_PROF") or "")
    if label and label != "nan":
        ax.annotate(label, (xs[0], ys[0]), fontsize=7)
ax.set_title(f"Шейп: 26 профилей съёмки 384092, СК-42 (EPSG:28479)")
ax.set_aspect("equal")

ax = axes[1]
for i, p in enumerate(polys):
    if i in removed_ids:
        ax.plot(p[:, 0], -p[:, 1], color="red", lw=1.4)
    else:
        ax.plot(p[:, 0], -p[:, 1], color="#cccccc", lw=0.7)
ax.set_title("Наша оцифровка листа 23: красное = кандидаты в профили "
             "(removed из crossing_filter), пиксели")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "profiles_shape_vs_digitized.png", bbox_inches="tight",
            facecolor="white")
print("saved:", OUT / "profiles_shape_vs_digitized.png")
