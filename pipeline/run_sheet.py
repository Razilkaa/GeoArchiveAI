"""Full-sheet label extraction: detect all text boxes on a map sheet,
export crops + JSONL manifest for VLM reading.

Usage: python run_sheet.py [path_to_sheet.jpg]
Output: pipeline/output/<sheet_name>/ {crops/*.png, manifest.jsonl, boxes_overview.png}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SRC = r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg"

# font-height bands from sheet statistics (stats.py): main mode 15-29 px = piket
# digits, 30-90 px = isoline labels / structure ids. Fill filter kills curve junk.
SMALL_H = (8, 29)
BIG_H = (30, 90)
BIG_FILL = 0.25
PAD = 8          # crop padding, px
UPSCALE = 2      # crop upscale factor for easier VLM reading


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def detect(img):
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    small = np.zeros_like(bw)
    big = np.zeros_like(bw)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        asp = max(w, h) / max(1, min(w, h))
        fill = a / max(1, w * h)
        if SMALL_H[0] <= h <= SMALL_H[1] and 3 <= w <= 60 and a >= 15 and asp < 5 and fill > 0.15:
            small[lab == i] = 255
        elif BIG_H[0] <= h <= BIG_H[1] and 8 <= w <= 90 and asp < 3 and fill >= BIG_FILL:
            big[lab == i] = 255

    def cluster(mask, dil, min_chars, wmin, hmin, wmax, hmax):
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
        blob = cv2.dilate(mask, kern)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if not (wmin <= w <= wmax and hmin <= h <= hmax):
                continue
            nn = cv2.connectedComponents(mask[y:y + h, x:x + w])[0] - 1
            if nn >= min_chars:
                out.append((x, y, w, h, nn))
        return out

    piket = cluster(small, 17, 3, 30, 18, 200, 200)
    iso = cluster(big, 29, 2, 40, 30, 300, 160)
    return piket, iso


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_SRC)
    out_dir = Path(__file__).parent / "output" / src.stem
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = imread_u(src)
    print("sheet:", img.shape)
    piket, iso = detect(img)
    print(f"boxes: piket={len(piket)} isoline={len(iso)}")

    boxes = [(x, y, w, h, nn, "small") for x, y, w, h, nn in piket] + \
            [(x, y, w, h, nn, "big") for x, y, w, h, nn in iso]
    boxes.sort(key=lambda b: (b[1] // 200, b[0]))

    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i, (x, y, w, h, nn, band) in enumerate(boxes):
            x0, y0 = max(0, x - PAD), max(0, y - PAD)
            x1, y1 = min(img.shape[1], x + w + PAD), min(img.shape[0], y + h + PAD)
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (crop.shape[1] * UPSCALE, crop.shape[0] * UPSCALE),
                              interpolation=cv2.INTER_LANCZOS4)
            name = f"box_{i:05d}.png"
            ok, buf = cv2.imencode(".png", crop)
            buf.tofile(str(crops_dir / name))
            f.write(json.dumps({
                "id": i, "crop": f"crops/{name}",
                "bbox": [int(x), int(y), int(w), int(h)],
                "center": [int(x + w / 2), int(y + h / 2)],
                "band": band, "n_chars": int(nn),
            }, ensure_ascii=False) + "\n")

    # overview with boxes
    scale = 6
    ov = cv2.cvtColor(cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale)),
                      cv2.COLOR_GRAY2BGR)
    for x, y, w, h, nn, band in boxes:
        col = (0, 0, 255) if band == "small" else (0, 170, 0)
        cv2.rectangle(ov, (x // scale, y // scale), ((x + w) // scale, (y + h) // scale), col, 1)
    cv2.imencode(".png", ov)[1].tofile(str(out_dir / "boxes_overview.png"))
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
