"""Topological rule: isolines of one horizon never cross each other.
Any crossing means one of the two curves is alien (profile/fault leak or bad
stitch). Iteratively remove the curve least similar to the bundle:
guilt = crossings x (1 + straightness-likeness + non-parallelism to neighbours).

Usage: python crossing_filter.py <sheet>   (e.g. 23)
Reads  out/<sheet>/isolines.json, writes out/<sheet>/isolines_clean.json
(polylines kept + removed list with reasons) and a review overlay.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point

Image.MAX_IMAGE_PIXELS = None
SHEET = sys.argv[1] if len(sys.argv) > 1 else "23"
TR = Path(r"C:\FINAM\Conference\pipeline\tracing\out") / SHEET
BMPS = {p.stem: p for p in
        Path(r"C:\FINAM\Conference\384092\Графика\Том 2, 3 (выборка)").glob("*.bmp")}

END_TOL = 12.0   # crossings this close to an endpoint = stitching junction, ignore


def straight_likeness(p):
    a, b = p[0], p[-1]
    chord = np.linalg.norm(b - a)
    if chord < 1:
        return 0.0
    ab = b - a
    dev = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / chord
    return float(max(0.0, 1.0 - (dev.max() / chord) / 0.15))  # 1 = perfectly straight


def local_tangents(p, k=10):
    t = np.gradient(p, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-6)


def main():
    data = json.loads((TR / "isolines.json").read_text(encoding="utf-8"))
    polys = [np.array(pl, float) for pl in data["polylines_xy"]]
    lines = [LineString(pl) for pl in polys]
    n = len(polys)
    print("curves:", n)

    # non-parallelism to neighbours (isolines run locally parallel)
    all_pts = np.vstack(polys)
    owner = np.concatenate([[i] * len(p) for i, p in enumerate(polys)])
    all_tan = np.vstack([local_tangents(p) for p in polys])
    tree = cKDTree(all_pts)
    nonpar = np.zeros(n)
    for i, p in enumerate(polys):
        samp = p[:: max(1, len(p) // 40)]
        tan = local_tangents(samp)
        devs = []
        for pt, tg in zip(samp, tan):
            for j in tree.query_ball_point(pt, 90):
                if owner[j] != i:
                    c = abs(float(np.dot(tg, all_tan[j])))
                    devs.append(1.0 - c)
                    break
        nonpar[i] = float(np.mean(devs)) if devs else 0.0

    def proper_crossings(i, j):
        inter = lines[i].intersection(lines[j])
        if inter.is_empty:
            return 0
        pts = []
        if isinstance(inter, Point):
            pts = [inter]
        elif isinstance(inter, MultiPoint):
            pts = list(inter.geoms)
        else:  # lines overlapping etc. - treat as heavy crossing
            return 3
        cnt = 0
        ends = [Point(polys[i][0]), Point(polys[i][-1]),
                Point(polys[j][0]), Point(polys[j][-1])]
        for pt in pts:
            if min(pt.distance(e) for e in ends) > END_TOL:
                cnt += 1
        return cnt

    cross = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if lines[i].envelope.intersects(lines[j].envelope):
                c = proper_crossings(i, j)
                cross[i, j] = cross[j, i] = c

    removed = []
    active = set(range(n))
    while True:
        counts = {i: int(cross[i, list(active - {i})].sum()) for i in active}
        bad = {i: c for i, c in counts.items() if c > 0}
        if not bad:
            break
        guilt = {i: c * (1.0 + straight_likeness(polys[i]) + nonpar[i])
                 for i, c in bad.items()}
        worst = max(guilt, key=guilt.get)
        removed.append({"id": worst, "crossings": bad[worst],
                        "straight_likeness": round(straight_likeness(polys[worst]), 2),
                        "non_parallelism": round(nonpar[worst], 2)})
        active.discard(worst)
    print(f"removed {len(removed)} crossing curves:",
          [(r["id"], r["crossings"], r["straight_likeness"]) for r in removed])

    out = {"source": data.get("source"), "n_polylines": len(active),
           "kept_ids": sorted(active),
           "removed": removed,
           "polylines_xy": [data["polylines_xy"][i] for i in sorted(active)],
           "orig_ids": sorted(active)}
    (TR / "isolines_clean.json").write_text(json.dumps(out, ensure_ascii=False),
                                            encoding="utf-8")

    # review overlay
    bmp = BMPS.get(SHEET)
    img = np.array(Image.open(bmp).convert("L")) if bmp else None
    if img is not None:
        vis = cv2.cvtColor(255 - ((img < 128).astype(np.uint8) * 80), cv2.COLOR_GRAY2BGR)
        rng = np.random.default_rng(3)
        for i in sorted(active):
            col = tuple(int(c) for c in rng.integers(30, 200, 3))
            cv2.polylines(vis, [polys[i].astype(np.int32)], False, col, 4)
        for r in removed:
            cv2.polylines(vis, [polys[r["id"]].astype(np.int32)], False, (0, 0, 255), 3)
        small = cv2.resize(vis, (img.shape[1] // 3, img.shape[0] // 3))
        cv2.imencode(".png", small)[1].tofile(str(TR / "crossing_filter.png"))
        print("overlay:", TR / "crossing_filter.png")


if __name__ == "__main__":
    main()
