"""Two-map scheme for sheet 23: map A = labeled isolines, map B = surface from
depth marks. Cross-check A vs B, propagate values to unlabeled curves from B
(with provenance flag), locate structure digits."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
TR = Path(r"C:\FINAM\Conference\pipeline\tracing\out\23")
DET = Path(r"C:\FINAM\Conference\pipeline\output_v4\23")
BMP = Path(r"C:\FINAM\Conference\384092\Графика\Том 2, 3 (выборка)\23.bmp")
STEP = 0.2

va = json.loads((TR / "valued_isolines_23.json").read_text(encoding="utf-8"))
iso_data = json.loads((TR / "isolines.json").read_text(encoding="utf-8"))
polylines = [np.array(p, float) for p in iso_data["polylines_xy"]]
readings = [json.loads(l) for l in (DET / "readings.jsonl").read_text(encoding="utf-8").splitlines()]
labeled = {int(k): v for k, v in va["values"].items()}
confident = set(va["confident"])
leaks = set(va["profile_leaks"])

# ---- map B: depth marks -> surface ----
RE_MARK = re.compile(r"^-?[0-9]\.[0-9]{2}$")
marks = []
for r in readings:
    if r.get("unreadable") or not r.get("values") or r["zone"] != "map_body":
        continue
    q = np.array(r["quad"])
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    for v in r["values"]:
        t = str(v.get("text", "")).strip().replace(",", ".").replace("−", "-")
        if RE_MARK.match(t):
            val = abs(float(t))
            if 1.2 <= val <= 5.8:
                marks.append((cx, cy, val))
marks = np.array(marks)
print("depth marks:", len(marks))

tree = cKDTree(marks[:, :2])
_, idx = tree.query(marks[:, :2], k=min(9, len(marks)))
med = np.median(marks[idx[:, 1:], 2], axis=1)
marks = marks[np.abs(marks[:, 2] - med) <= 0.25]
print("after outlier filter:", len(marks))
rbf = RBFInterpolator(marks[:, :2], marks[:, 2], kernel="thin_plate_spline",
                      smoothing=1.0, neighbors=40)
mtree = cKDTree(marks[:, :2])

# ---- sample surface along every curve ----
def surface_value(p):
    samp = p[:: max(1, len(p) // 60)]
    d, _ = mtree.query(samp)
    inside = samp[d < 350]
    if len(inside) < 8:
        return None, None, 0.0
    vals = rbf(inside)
    return float(np.median(vals)), float(np.std(vals)), len(inside) / len(samp)


from_surface = {}
surf_check = {}
verdict = {}   # main / combo / flagged
for i, p in enumerate(polylines):
    if i in leaks:
        continue
    sv, sstd, cover = surface_value(p)
    if sv is None:
        if i in labeled:
            verdict[i] = "combo"   # подпись есть, поверхность не покрывает
        continue
    snapped = round(round(sv / STEP) * STEP, 2)
    surf_check[i] = {"surface": round(sv, 2), "snapped": -abs(snapped),
                     "std": round(sstd, 2), "coverage": round(cover, 2)}
    if i in labeled:
        diff_steps = abs(abs(labeled[i]) - sv) / STEP
        if diff_steps <= 1.0:
            verdict[i] = "main"     # подпись и поверхность согласны
        elif diff_steps <= 1.5:
            verdict[i] = "combo"
        else:
            verdict[i] = "flagged"  # чужая подпись или неверная сшивка
    elif sstd <= 0.13 and cover >= 0.55:
        from_surface[i] = -abs(snapped)
        verdict[i] = "combo"
from collections import Counter as _C
print("verdicts:", dict(_C(verdict.values())))
print("gray curves valued from surface:", len(from_surface))

# ---- structure digits (loosened) ----
structs = []
for r in readings:
    if r.get("unreadable") or not r.get("values") or r["zone"] != "map_body":
        continue
    vals = [str(v.get("text", "")).strip() for v in r["values"]]
    if len(vals) == 1 and vals[0] in ("1", "2"):
        q = np.array(r["quad"])
        # isolated: far from depth marks cloud
        d, _ = mtree.query([q[:, 0].mean(), q[:, 1].mean()])
        structs.append((q[:, 0].mean(), q[:, 1].mean(), vals[0], float(d)))
structs = [s for s in structs if s[3] > 60]
print("structure digit candidates:", [(t, int(x), int(y), int(d)) for x, y, t, d in structs])

# ---- combined render ----
img = np.array(Image.open(BMP).convert("L"))
H, W = img.shape
fig, ax = plt.subplots(figsize=(16, 15), dpi=110)
ax.imshow(img, cmap="gray", alpha=0.25)
cmap = plt.get_cmap("turbo")
allv = [abs(v) for v in labeled.values()] + [abs(v) for v in from_surface.values()]
vmin, vmax = min(allv), max(allv)


def col_of(v):
    return cmap((vmax - abs(v)) / max(0.1, vmax - vmin))  # свод тёплым (геол. конвенция)


for i, p in enumerate(polylines):
    vd = verdict.get(i)
    if i in leaks:
        ax.plot(p[:, 0], p[:, 1], color="#88aacc", lw=0.7, ls=":", alpha=0.6)
    elif vd == "flagged":
        ax.plot(p[:, 0], p[:, 1], color="red", lw=2.2, ls="--")
        mid = p[len(p) // 2]
        ax.annotate(f"?{labeled.get(i, '')}", mid, fontsize=8, color="red",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ffecec", ec="red"))
    elif i in labeled:
        v = labeled[i]
        ax.plot(p[:, 0], p[:, 1], color=col_of(v),
                lw=2.6 if vd == "main" else 1.7, ls="-" if vd == "main" else "--")
        mid = p[len(p) // 2]
        ax.annotate(f"{v}", mid, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=col_of(v), lw=1.2))
    elif i in from_surface:
        v = from_surface[i]
        ax.plot(p[:, 0], p[:, 1], color=col_of(v), lw=1.4, ls="-.", alpha=0.9)
        mid = p[len(p) // 2]
        ax.annotate(f"{v}*", mid, fontsize=6, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#f2f2f2",
                              ec=col_of(v), lw=0.8))
    else:
        ax.plot(p[:, 0], p[:, 1], color="#aaaaaa", lw=0.9, alpha=0.7)
for x, y, t, d in structs:
    ax.scatter(x, y, s=420, facecolors="none", edgecolors="magenta", linewidths=2.5)
    name = {"1": "1: Шеинская", "2": "2: Зап.-Шеинская"}[t]
    ax.annotate(name, (x, y), xytext=(14, -16), textcoords="offset points",
                fontsize=11, color="magenta", weight="bold")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
ax.set_title(
    "Лист 23: двухкарточная схема. Сплошные - подпись+поверхность согласны; "
    "пунктир - только подпись (конфликт); штрих-пунктир со * - значение от "
    "поверхности отметок; точечные - профили", fontsize=10)
fig.tight_layout()
fig.savefig(TR / "combined_23.png", bbox_inches="tight", facecolor="white")

out = {"verdicts": {str(k): v for k, v in verdict.items()}, "from_surface": {str(k): v for k, v in from_surface.items()},
       "surface_check": {str(k): v for k, v in surf_check.items()},
       "structures": [{"digit": t, "x": x, "y": y} for x, y, t, d in structs]}
(TR / "crosscheck_23.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
print("saved:", TR / "combined_23.png")
