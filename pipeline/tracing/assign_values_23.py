"""Sheet 23 showcase: assign read values to traced isolines, locate structures,
check the closure claim from the report text (Шеинская подготовлена по -3800 м)."""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
TR = Path(r"C:\FINAM\Conference\pipeline\tracing\out\23")
DET = Path(r"C:\FINAM\Conference\pipeline\output_v4\23")
BMP = Path(r"C:\FINAM\Conference\384092\Графика\Том 2, 3 (выборка)\23.bmp")

RE_ISO = re.compile(r"^-[0-9]\.[0-9]{1,2}$")
RE_STRUCT = re.compile(r"^[12]$")

CONTOUR_STEP = 0.2   # km; TODO: брать из текста отчёта (раздел о картопостроении) через RAG
SNAP_TOL = 0.06      # km: label deviating from step-multiple less than this -> snap

iso_data = json.loads((TR / "isolines.json").read_text(encoding="utf-8"))
polylines = [np.array(p, float) for p in iso_data["polylines_xy"]]
readings = [json.loads(l) for l in (DET / "readings.jsonl").read_text(encoding="utf-8").splitlines()]
print("polylines:", len(polylines), " readings:", len(readings))

# --- drop profile leaks: isolines are sinuous, profiles are straight ---
def straightness(p):
    """max deviation from chord / chord length; near 0 => straight line"""
    a, b = p[0], p[-1]
    chord = np.linalg.norm(b - a)
    if chord < 1:
        return 0.0
    ab = b - a
    d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / chord
    return float(d.max() / chord)

is_profile_leak = []
for p in polylines:
    length = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    is_profile_leak.append(straightness(p) < 0.035 and length > 400)
print("profile leaks excluded:", sum(is_profile_leak))

# label candidates
iso_labels, struct_marks = [], []
for r in readings:
    if r.get("unreadable") or not r.get("values") or r["zone"] != "map_body":
        continue
    q = np.array(r["quad"])
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    for v in r["values"]:
        t = str(v.get("text", "")).strip().replace(",", ".").replace("−", "-")
        if RE_ISO.match(t):
            iso_labels.append((cx, cy, float(t)))
        elif RE_STRUCT.match(t) and r["font_band"] in ("big", "unknown"):
            struct_marks.append((cx, cy, t))
print("isoline labels:", len(iso_labels), " structure digit marks:", len(struct_marks))

# contour-step rule: isoline labels are step-multiples; snap or reject
def normalize_label(v):
    snapped = round(v / CONTOUR_STEP) * CONTOUR_STEP
    if abs(v - snapped) <= SNAP_TOL:
        return round(snapped, 2), True
    return v, False   # некратное сечению = скорее отметка глубины, не изогипса

iso_labels_ok = []
rejected = 0
for cx, cy, val in iso_labels:
    nv, ok = normalize_label(val)
    if ok:
        iso_labels_ok.append((cx, cy, nv))
    else:
        rejected += 1
print(f"labels: {len(iso_labels_ok)} step-multiple, {rejected} rejected (not multiple of {CONTOUR_STEP})")

# assign labels to nearest polyline (label sits in its own curve's gap)
trees = [cKDTree(p) for p in polylines]
votes = {}
for cx, cy, val in iso_labels_ok:
    best_i, best_d = None, 70.0
    for i, t in enumerate(trees):
        if is_profile_leak[i]:
            continue
        d, _ = t.query([cx, cy])
        if d < best_d:
            best_i, best_d = i, d
    if best_i is not None:
        votes.setdefault(best_i, []).append(val)

valued = {}
confident = set()
conflicts = 0
for i, vs in votes.items():
    c = Counter(vs)
    val, cnt = c.most_common(1)[0]
    if len(c) == 1 or cnt >= 2 * (len(vs) - cnt):
        confident.add(i)   # единогласно или уверенное большинство
    else:
        conflicts += 1
    valued[i] = val
print(f"polylines with value: {len(valued)}/{len(polylines)} "
      f"(confident: {len(confident)}), conflict: {conflicts}")

# closure check: closed polyline = endpoints near vs length
closures = []
for i, val in valued.items():
    p = polylines[i]
    if len(p) < 50:
        continue
    end_gap = np.linalg.norm(p[0] - p[-1])
    length = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    if end_gap < 0.12 * length and length > 800:
        closures.append((i, val, end_gap, length))
print("closed valued isolines:", [(i, v, int(l)) for i, v, g, l in closures])

# render
img = np.array(Image.open(BMP).convert("L"))
H, W = img.shape
fig, ax = plt.subplots(figsize=(16, 15), dpi=110)
ax.imshow(img, cmap="gray", alpha=0.30)
vals = sorted({v for v in valued.values()})
cmap = plt.get_cmap("turbo")
vmin, vmax = min(vals) if vals else -5, max(vals) if vals else -1
for i, p in enumerate(polylines):
    if is_profile_leak[i]:
        ax.plot(p[:, 0], p[:, 1], color="#6688bb", lw=0.8, ls=":", alpha=0.7)
    elif i in valued:
        col = cmap((valued[i] - vmin) / max(0.1, vmax - vmin))
        lw = 2.4 if i in confident else 1.6
        ls = "-" if i in confident else "--"
        ax.plot(p[:, 0], p[:, 1], color=col, lw=lw, ls=ls)
        mid = p[len(p) // 2]
        ax.annotate(f"{valued[i]}", mid, fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1))
    else:
        ax.plot(p[:, 0], p[:, 1], color="#999999", lw=1.0, alpha=0.8)
for cx, cy, t in struct_marks:
    ax.scatter(cx, cy, s=380, facecolors="none", edgecolors="magenta", linewidths=2.5)
    name = {"1": "1: Шеинская", "2": "2: Западно-Шеинская"}[t]
    ax.annotate(name, (cx, cy), xytext=(12, -14), textcoords="offset points",
                fontsize=11, color="magenta", weight="bold")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
ax.set_title("Лист 23 (Шеинская площадь): трассированные изогипсы с автоматически "
             "присвоенными значениями\nСерые — значение не присвоено. "
             "Текст отчёта: «подготовлена по замыкающейся изогипсе -3800 м»", fontsize=11)
fig.tight_layout()
fig.savefig(TR / "valued_isolines_23.png", bbox_inches="tight", facecolor="white")

out = {"n_polylines": len(polylines), "n_valued": len(valued),
       "contour_step_km": CONTOUR_STEP,
       "confident": sorted(confident),
       "profile_leaks": [i for i, f in enumerate(is_profile_leak) if f],
       "values": {str(i): v for i, v in valued.items()},
       "closures": [{"polyline": i, "value": v, "length_px": int(l)} for i, v, g, l in closures],
       "structures": [{"digit": t, "x": cx, "y": cy} for cx, cy, t in struct_marks]}
(TR / "valued_isolines_23.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
print("saved:", TR / "valued_isolines_23.png")
