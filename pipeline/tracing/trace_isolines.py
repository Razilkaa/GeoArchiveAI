"""Isoline tracing v1 on 1-bit archival scans (recall-oriented).

Idea: text/ticks/hatching are SMALL connected components -> drop by size;
straight profiles/frame -> Hough erase; what remains are the authored curves.
Skeletonize and walk into polylines.

Usage: python trace_isolines.py [path_to_1bit_scan]
Output: pipeline/tracing/out/<name>/ {isolines.json, overlay.png, stats}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

DEFAULT = r"C:\FINAM\Conference\384092\Графика\Том 2, 3 (выборка)\21.bmp"

MIN_COMP = 110        # px: components smaller than this = text/ticks -> dropped
                      # (chars are <=90; 110 keeps mid-size curve fragments)
LINE_MIN_LEN = 500    # Hough: straight segments (profiles, frame)
LINE_ERASE_W = 9
BORDER = 60           # px: sheet border zone erased
MIN_POLYLINE = 150    # px: shorter traced paths dropped
SIMPLIFY_EPS = 2.0


def imread_1bit(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("L")
    return np.array(im)


def extract_polylines(skel):
    """Walk skeleton into paths between endpoints/junctions."""
    h, w = skel.shape
    pts = np.argwhere(skel)
    on = set(map(tuple, pts))
    nbrs_off = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbors(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in nbrs_off
                if (p[0] + dy, p[1] + dx) in on]

    deg = {p: len(neighbors(p)) for p in on}
    nodes = {p for p, d in deg.items() if d != 2}
    visited_edges = set()
    polylines = []

    def walk(start, first):
        path = [start, first]
        prev, cur = start, first
        while cur not in nodes:
            nxt = [n for n in neighbors(cur) if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            if len(path) > 500000:
                break
        return path

    for node in nodes:
        for nb in neighbors(node):
            ek = (node, nb)
            if ek in visited_edges:
                continue
            path = walk(node, nb)
            visited_edges.add((path[0], path[1]))
            visited_edges.add((path[-1], path[-2]) if len(path) > 1 else ek)
            if len(path) >= 2:
                polylines.append(path)

    # pure loops (no junction on them)
    seen = set()
    for pl in polylines:
        seen.update(pl)
    leftover = on - seen
    while leftover:
        start = next(iter(leftover))
        nb = [n for n in neighbors(start) if n in leftover]
        if not nb:
            leftover.discard(start)
            continue
        path = [start]
        prev, cur = start, nb[0]
        while cur != start and cur in leftover:
            path.append(cur)
            nxt = [n for n in neighbors(cur) if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        polylines.append(path)
        leftover -= set(path)
    return polylines


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
    out_dir = Path(__file__).parent / "out" / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    g = imread_1bit(src)
    ink = (g < 128).astype(np.uint8)
    H, W = ink.shape
    print("sheet:", ink.shape)

    # 1) split components: long (solid curves) / dash-like (dashed curves) / junk
    n, lab, stats, cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros_like(ink)
    dashes = []  # (cx, cy, ux, uy, half_len, comp_id)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        mx, mn = max(w, h), min(w, h)
        if mx >= MIN_COMP:
            keep[lab == i] = 1
        elif 14 <= mx <= 150 and a >= 12 and mx / max(1, mn) >= 2.2:
            ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
            pts_ = np.column_stack([xs + x, ys + y]).astype(float)
            mean = pts_.mean(0)
            _, sv, vt = np.linalg.svd(pts_ - mean)
            thin = sv[1] / max(1e-3, sv[0])
            if thin < 0.35:  # thin stroke
                dashes.append((mean[0], mean[1], vt[0][0], vt[0][1], mx / 2, i))
    print(f"components: {n - 1}, solid-long kept, dash candidates: {len(dashes)}")

    # 1b) chain dashes whose own orientation continues the chain direction
    if dashes:
        from scipy.spatial import cKDTree
        dc = np.array([(d[0], d[1]) for d in dashes])
        du = np.array([(d[2], d[3]) for d in dashes])
        dl = np.array([d[4] for d in dashes])
        tree = cKDTree(dc)
        links = 0
        for i in range(len(dashes)):
            for j in tree.query_ball_point(dc[i], 80):
                if j <= i:
                    continue
                v = dc[j] - dc[i]
                dist = np.linalg.norm(v)
                if dist < 4 or dist > dl[i] + dl[j] + 65:
                    continue
                vn = v / dist
                # link direction must align with BOTH dash orientations
                if abs(float(np.dot(vn, du[i]))) < 0.9 or abs(float(np.dot(vn, du[j]))) < 0.9:
                    continue
                cv2.line(keep, tuple(dc[i].astype(int)), tuple(dc[j].astype(int)), 1, 3)
                for d in (i, j):
                    keep[lab == dashes[d][5]] = 1
                links += 1
        print(f"dash links drawn: {links}")

    # 2) erase straight lines (profiles, frame)
    segs = cv2.HoughLinesP(keep * 255, 1, np.pi / 360, threshold=300,
                           minLineLength=LINE_MIN_LEN, maxLineGap=12)
    segs = segs.reshape(-1, 4) if segs is not None else []
    for x1, y1, x2, y2 in segs:
        cv2.line(keep, (x1, y1), (x2, y2), 0, LINE_ERASE_W)
    print("straight segments erased:", len(segs))
    keep[:BORDER] = keep[-BORDER:] = 0
    keep[:, :BORDER] = keep[:, -BORDER:] = 0

    # drop crumbs created by erasing
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    keep2 = np.zeros_like(keep)
    for i in range(1, n2):
        if max(st2[i, 2], st2[i, 3]) >= 100:
            keep2[lab2 == i] = 1

    # 3) skeletonize + walk
    skel = skeletonize(keep2 > 0)
    print("skeleton px:", int(skel.sum()))
    polylines = extract_polylines(skel)
    polylines = [p for p in polylines if len(p) >= 25]
    print("raw paths:", len(polylines))

    # 3b) stitch through junctions by tangent continuity
    def endpoints(p):
        return p[0], p[-1]

    def tangent(p, at_start, k=12):
        seg = p[:k] if at_start else p[-k:][::-1]
        (y0, x0), (y1, x1) = seg[-1], seg[0]
        v = np.array([x1 - x0, y1 - y0], float)
        n = np.linalg.norm(v)
        return v / n if n else v

    def stitch(paths, max_gap=6, min_cos=0.82):
        changed = True
        while changed:
            changed = False
            ends = []  # (pt, path_idx, is_start)
            for i, p in enumerate(paths):
                if p is None:
                    continue
                a, b = endpoints(p)
                ends.append((a, i, True))
                ends.append((b, i, False))
            used = set()
            for e1 in range(len(ends)):
                if e1 in used:
                    continue
                p1, i1, s1 = ends[e1]
                best, best_cos = None, min_cos
                for e2 in range(e1 + 1, len(ends)):
                    if e2 in used:
                        continue
                    p2, i2, s2 = ends[e2]
                    if i1 == i2 or paths[i1] is None or paths[i2] is None:
                        continue
                    if abs(p1[0] - p2[0]) > max_gap or abs(p1[1] - p2[1]) > max_gap:
                        continue
                    t1 = tangent(paths[i1], s1)
                    t2 = tangent(paths[i2], s2)
                    bridge = np.array([p2[1] - p1[1], p2[0] - p1[0]], float)
                    bn = np.linalg.norm(bridge)
                    if bn > 3:  # bridge must continue BOTH tangents, not go sideways
                        bridge /= bn
                        if float(np.dot(t1, bridge)) < best_cos or \
                           float(np.dot(-t2, bridge)) < best_cos:
                            continue
                    c = -float(np.dot(t1, t2))  # continuation = opposite tangents
                    if c > best_cos:
                        best, best_cos = e2, c
                if best is None:
                    continue
                p2, i2, s2 = ends[best]
                a = paths[i1] if not s1 else paths[i1][::-1]   # end at junction
                b = paths[i2] if s2 else paths[i2][::-1]       # start at junction
                paths[i1] = a + b
                paths[i2] = None
                used.add(e1)
                used.add(best)
                changed = True
        return [p for p in paths if p is not None]

    polylines = stitch(list(polylines))                          # junction gaps
    polylines = stitch(polylines, max_gap=45, min_cos=0.93)      # erased-line nicks
    polylines = stitch(polylines, max_gap=130, min_cos=0.965)    # label gaps
    polylines = stitch(polylines, max_gap=210, min_cos=0.985)    # long tangent jumps
    polylines = [p for p in polylines if len(p) >= MIN_POLYLINE]
    print("after stitching, >= min length:", len(polylines))

    # simplify, save
    simplified = []
    for p in polylines:
        arr = np.array([(x, y) for y, x in p], dtype=np.int32).reshape(-1, 1, 2)
        ap = cv2.approxPolyDP(arr, SIMPLIFY_EPS, False).reshape(-1, 2)
        simplified.append(ap.tolist())
    (out_dir / "isolines.json").write_text(json.dumps(
        {"source": str(src), "n_polylines": len(simplified),
         "polylines_xy": simplified}, ensure_ascii=False), encoding="utf-8")

    # overlay: faded original + colored traces
    vis = cv2.cvtColor(255 - (ink * 90), cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(1)
    for pl in simplified:
        col = tuple(int(c) for c in rng.integers(40, 230, 3))
        cv2.polylines(vis, [np.array(pl, np.int32)], False, col, 4)
    small = cv2.resize(vis, (W // 3, H // 3))
    cv2.imencode(".png", small)[1].tofile(str(out_dir / "overlay.png"))
    total_len = sum(len(p) for p in polylines)
    print(f"traced total ~{total_len} px of curves -> {out_dir}")


if __name__ == "__main__":
    main()
