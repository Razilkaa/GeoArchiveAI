"""Detector v4: oriented text-group detection on map sheets.

Addresses v3 problems:
  1. axis-aligned dilation merging neighbours  -> graph grouping with per-char radius
  2. lost minus / decimal dots                 -> mark recovery inside expanded oriented rect
  3. profile lines detected as symbols         -> line mask separated by elongation
  4. multiple values in one bbox               -> split on large inter-symbol gap along baseline
  5. legend / title block junk                 -> zone tagging (map_body/legend/title_block/frame)
  6. unstable small/big bands on rotated text  -> band from char max-dimension, 'unknown' if mixed

No OCR, no reading, no geological classes. Output: manifest_v4.jsonl + crops + overlay.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")
OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
CROPS = OUT / "crops"
CTX = OUT / "context_crops"

# --- char / mark / line component filters ---
CHAR_SIZE = (10, 90)        # max dimension
CHAR_MIN_DIM = 4
CHAR_ASPECT = 5.0
CHAR_FILL = 0.18            # v4.1: was 0.12, curve fragments leaked into groups
CHAR_AREA = 25
LINE_LEN = 120              # longer than any single character on this sheet
LINE_ASPECT = 6.0
# grouping (v4.1: tightened after 100-box validation - merged was 51%)
LINK_K = 1.05               # link if dist < LINK_K*(s_i+s_j)/2 + 3
LINK_SIZE_RATIO = 1.8
SPLIT_GAP_K = 1.25          # split group when projected gap > K * median char size
ROW_SPLIT_K = 0.7           # transverse split: gap across baseline > K * median size
DASH_CHAIN_ASPECT = 2.3     # group where >=80% comps are this elongated = tick chain
EXT_K = 1.2                 # baseline extension (char sizes) for mark recovery
SMALL_MAX = 30              # font bands by char max-dimension
BIG_MIN = 31
PAD_CROP = 6
PAD_CTX = 40
UPSCALE = 2
# zones (sheet 21 layout, px)
FRAME_MARGIN = 120
TITLE_X, TITLE_Y = 6450, 5900
LEGEND_Y = 5900


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite_u(path, img):
    cv2.imencode(Path(path).suffix, img)[1].tofile(str(path))


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def norm_rect(rect):
    """minAreaRect -> (cx, cy, major, minor, angle_deg of major axis in [-90, 90))"""
    (cx, cy), (w, h), ang = rect
    if w < h:
        w, h = h, w
        ang += 90.0
    while ang >= 90:
        ang -= 180
    while ang < -90:
        ang += 180
    return cx, cy, w, h, ang


def quad_of(cx, cy, major, minor, ang):
    pts = cv2.boxPoints(((cx, cy), (major, minor), ang))
    return pts


def zone_of(cx, cy, W, H):
    frame_margin = max(40, int(min(W, H) * 0.015))
    if cx < frame_margin or cy < frame_margin or cx > W - frame_margin or cy > H - frame_margin:
        return "frame"
    if cx >= W * 0.78 and cy >= H * 0.82:
        return "title_block"
    if cy >= H * 0.84:
        return "legend"
    return "map_body"


def main(src=SRC, out=OUT):
    global SRC, OUT, CROPS, CTX
    SRC = Path(src)
    OUT = Path(out)
    CROPS = OUT / "crops"
    CTX = OUT / "context_crops"
    CROPS.mkdir(parents=True, exist_ok=True)
    CTX.mkdir(parents=True, exist_ok=True)
    img = imread_u(SRC)
    H, W = img.shape
    print("sheet:", img.shape)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)

    chars, marks = [], []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        mx, mn = max(w, h), min(w, h)
        asp = mx / max(1, mn)
        fill = a / max(1, w * h)
        if mx >= LINE_LEN or (asp >= LINE_ASPECT and mx > 40):
            continue  # long line / fault hatching backbone
        if CHAR_SIZE[0] <= mx <= CHAR_SIZE[1] and mn >= CHAR_MIN_DIM and asp < CHAR_ASPECT \
                and fill > CHAR_FILL and a >= CHAR_AREA:
            chars.append((i, x, y, w, h, a, cent[i][0], cent[i][1], mx, fill))
        elif a >= 8 and mx <= 40 and (asp >= 2.0 or mx <= 12):
            marks.append((i, x, y, w, h, cent[i][0], cent[i][1]))  # dash / dot candidates
    print(f"char comps: {len(chars)}, mark comps: {len(marks)}")

    # --- grouping along arbitrary angle (graph on centers) ---
    pts = np.array([(c[6], c[7]) for c in chars])
    sizes = np.array([c[8] for c in chars], dtype=float)
    tree = cKDTree(pts)
    uf = UF(len(chars))
    for i in range(len(chars)):
        r = LINK_K * sizes[i] + 3
        for j in tree.query_ball_point(pts[i], r * 1.5):
            if j <= i:
                continue
            d = math.hypot(*(pts[i] - pts[j]))
            thr = LINK_K * (sizes[i] + sizes[j]) / 2 + 3
            ratio = max(sizes[i], sizes[j]) / max(1e-3, min(sizes[i], sizes[j]))
            if d < thr and ratio < LINK_SIZE_RATIO:
                uf.union(i, j)

    groups = {}
    for i in range(len(chars)):
        groups.setdefault(uf.find(i), []).append(i)
    groups = [g for g in groups.values() if len(g) >= 2]
    print("raw groups:", len(groups))

    # --- split groups: across baseline (stacked value columns), then along it ---
    def split_1d(g, axis_vec, gap_thr):
        p = pts[g]
        mean = p.mean(0)
        proj = (p - mean) @ axis_vec
        order = np.argsort(proj)
        parts, cur = [], [g[order[0]]]
        for a, b in zip(order[:-1], order[1:]):
            if proj[b] - proj[a] > gap_thr:
                parts.append(cur)
                cur = []
            cur.append(g[b])
        parts.append(cur)
        return [q for q in parts if q]

    def split_group(g):
        if len(g) < 3:
            return [g]
        p = pts[g]
        mean = p.mean(0)
        _, _, vt = np.linalg.svd(p - mean)
        med = float(np.median(sizes[g]))
        rows = split_1d(g, vt[1], max(6.0, ROW_SPLIT_K * med))      # transverse first
        out = []
        for row in rows:
            if len(row) < 3:
                out.append(row)
                continue
            pr = pts[row]
            _, _, vtr = np.linalg.svd(pr - pr.mean(0))
            medr = float(np.median(sizes[row]))
            out += split_1d(row, vtr[0], SPLIT_GAP_K * medr + 4)    # along baseline
        return out

    # iterate: splitting changes geometry, second pass catches leftovers
    for _ in range(2):
        groups = [q for g in groups for q in split_group(g)]
    groups = [g for g in groups if len(g) >= 2]
    print("after split:", len(groups))

    # --- drop tick / dash chains: nearly all components elongated the same way ---
    def is_dash_chain(g):
        asps = []
        for idx in g:
            _, x, y, w, h, a, cx, cy, mx, fill = chars[idx]
            asps.append(max(w, h) / max(1, min(w, h)))
        share = sum(1 for a_ in asps if a_ >= DASH_CHAIN_ASPECT) / len(asps)
        return share >= 0.8 and len(g) >= 3

    before = len(groups)
    groups = [g for g in groups if not is_dash_chain(g)]
    print(f"dash chains dropped: {before - len(groups)}")

    # --- build oriented rects, recover marks, score ---
    mark_pts = np.array([(m[5], m[6]) for m in marks]) if marks else np.zeros((0, 2))
    mark_tree = cKDTree(mark_pts) if len(mark_pts) else None
    records = []
    for g in groups:
        corner_pts = []
        for idx in g:
            _, x, y, w, h, a, cx, cy, mx, fill = chars[idx]
            corner_pts += [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        rect = cv2.minAreaRect(np.array(corner_pts, dtype=np.float32))
        cx, cy, major, minor, ang = norm_rect(rect)
        med = float(np.median(sizes[g]))

        # recover dashes/dots inside rect extended along baseline
        ext = EXT_K * med
        n_marks = 0
        if mark_tree is not None:
            cand = mark_tree.query_ball_point([cx, cy], (major / 2 + ext) + 40)
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            for mi in cand:
                dx, dy = mark_pts[mi] - (cx, cy)
                lx = dx * ca + dy * sa
                ly = -dx * sa + dy * ca
                if abs(lx) <= major / 2 + ext and abs(ly) <= minor / 2 + 0.25 * med:
                    m = marks[mi]
                    corner_pts += [(m[1], m[2]), (m[1] + m[3], m[2] + m[4])]
                    n_marks += 1
        if n_marks:
            rect = cv2.minAreaRect(np.array(corner_pts, dtype=np.float32))
            cx, cy, major, minor, ang = norm_rect(rect)

        # confidence: alignment residual + spacing regularity + fill sanity + count
        p = pts[g]
        mean = p.mean(0)
        _, svals, vt = np.linalg.svd(p - mean)
        resid = (svals[1] / max(1, len(g) - 1) ** 0.5) / med if len(g) > 2 else 0.3
        proj = np.sort((p - mean) @ vt[0])
        gaps = np.diff(proj)
        cv_gap = float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 1 and np.mean(gaps) > 0 else 0.5
        mean_fill = float(np.mean([chars[i][9] for i in g]))
        conf = 0.3
        conf += 0.2 if len(g) >= 3 else 0.05
        conf += 0.2 * (1 - min(1.0, resid * 3))
        conf += 0.15 * (1 - min(1.0, cv_gap))
        conf += 0.15 if 0.18 <= mean_fill <= 0.6 else 0.0
        conf = round(min(1.0, conf), 2)

        band_sizes = sizes[g]
        if np.all(band_sizes <= SMALL_MAX):
            band = "small"
        elif np.all(band_sizes >= BIG_MIN):
            band = "big"
        else:
            band = "unknown"

        records.append({
            "cx": cx, "cy": cy, "major": major, "minor": minor, "angle": ang,
            "n": len(g) + n_marks, "band": band, "conf": conf, "med": med,
        })

    # --- oriented NMS ---
    def rot_inter(a, b):
        ra = ((a["cx"], a["cy"]), (a["major"], a["minor"]), a["angle"])
        rb = ((b["cx"], b["cy"]), (b["major"], b["minor"]), b["angle"])
        ok, region = cv2.rotatedRectangleIntersection(ra, rb)
        if region is None or ok == cv2.INTERSECT_NONE:
            return 0.0
        return cv2.contourArea(region)

    records.sort(key=lambda r: -r["conf"])
    kept = []
    centers = np.array([(r["cx"], r["cy"]) for r in records])
    rt = cKDTree(centers)
    dead = set()
    for i, r in enumerate(records):
        if i in dead:
            continue
        kept.append(r)
        for j in rt.query_ball_point(centers[i], r["major"]):
            if j <= i or j in dead:
                continue
            q = records[j]
            inter = rot_inter(r, q)
            area_r = r["major"] * r["minor"]
            area_q = q["major"] * q["minor"]
            iou = inter / max(1.0, area_r + area_q - inter)
            contain = inter / max(1.0, min(area_r, area_q))
            if iou > 0.5 or contain > 0.8:
                dead.add(j)
    print("after NMS:", len(kept))

    # --- manifest + crops + overlay ---
    kept.sort(key=lambda r: (int(r["cy"]) // 200, int(r["cx"])))
    scale = 6
    ov = cv2.cvtColor(cv2.resize(img, (W // scale, H // scale)), cv2.COLOR_GRAY2BGR)
    zone_col = {"map_body": None, "legend": (0, 140, 255), "title_block": (200, 0, 200),
                "frame": (140, 140, 140)}
    with open(OUT / "manifest_v4.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            quad = quad_of(r["cx"], r["cy"], r["major"] + 2 * PAD_CROP,
                           r["minor"] + 2 * PAD_CROP, r["angle"])
            xs, ys = quad[:, 0], quad[:, 1]
            x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
            x1, y1 = min(W, int(xs.max())), min(H, int(ys.max()))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            c2 = cv2.resize(crop, (crop.shape[1] * UPSCALE, crop.shape[0] * UPSCALE),
                            interpolation=cv2.INTER_LANCZOS4)
            name = f"box_{i:05d}.png"
            imwrite_u(CROPS / name, c2)
            cx0, cy0 = max(0, x0 - PAD_CTX), max(0, y0 - PAD_CTX)
            cx1, cy1 = min(W, x1 + PAD_CTX), min(H, y1 + PAD_CTX)
            imwrite_u(CTX / name, img[cy0:cy1, cx0:cx1])

            zone = zone_of(r["cx"], r["cy"], W, H)
            f.write(json.dumps({
                "id": i,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "quad": [[round(float(px), 1), round(float(py), 1)] for px, py in quad],
                "angle": round(float(r["angle"]), 1),
                "font_band": r["band"],
                "n_components": int(r["n"]),
                "zone": zone,
                "detector_confidence": r["conf"],
                "crop": f"crops/{name}",
                "context_crop": f"context_crops/{name}",
            }, ensure_ascii=False) + "\n")

            col = zone_col[zone] or ((0, 0, 255) if r["band"] == "small"
                                     else (0, 170, 0) if r["band"] == "big" else (255, 0, 0))
            q = (quad / scale).astype(np.int32)
            cv2.polylines(ov, [q], True, col, 1)

    imwrite_u(OUT / "overlay_v4.png", ov)
    print("manifest:", OUT / "manifest_v4.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect oriented text groups on a scanned map")
    parser.add_argument("src", nargs="?", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.src, args.out)
