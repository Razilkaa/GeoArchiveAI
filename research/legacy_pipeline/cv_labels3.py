"""v3: separate masks per font band BEFORE clustering; fill filter kills curve fragments."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad"
TILE = SCRATCH + r"\tile_center.png"

img = cv2.imread(TILE, cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)

small = np.zeros_like(bw)
big = np.zeros_like(bw)
for i in range(1, n):
    x, y, w, h, a = stats[i]
    asp = max(w, h) / max(1, min(w, h))
    fill = a / max(1, w * h)
    if 8 <= h <= 29 and 3 <= w <= 60 and a >= 15 and asp < 5 and fill > 0.15:
        small[lab == i] = 255
    elif 30 <= h <= 90 and 8 <= w <= 90 and asp < 3 and fill >= 0.25:
        big[lab == i] = 255

def cluster(mask, dil, min_chars, wmin, hmin, wmax, hmax):
    blob = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (wmin <= w <= wmax and hmin <= h <= hmax):
            continue
        sub = mask[y:y + h, x:x + w]
        nn = cv2.connectedComponents(sub)[0] - 1
        if nn >= min_chars:
            out.append((x, y, w, h))
    return out

piket = cluster(small, 17, 3, 30, 18, 160, 160)
iso = cluster(big, 29, 2, 40, 30, 300, 160)
print("piket:", len(piket), " isoline:", len(iso))

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y, w, h in piket:
    cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 2)
for k, (x, y, w, h) in enumerate(iso):
    cv2.rectangle(color, (x, y), (x + w, y + h), (0, 170, 0), 3)
    cv2.putText(color, f"i{k}", (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 0), 2)
cv2.imwrite(SCRATCH + r"\tile_overlay_v3.png", color)

src = Image.open(TILE)
pad = 10
cell_w, cell_h = 320, 140
cols = 4
rows = max(1, (len(iso) + cols - 1) // cols)
mont = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
d = ImageDraw.Draw(mont)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    font = ImageFont.load_default()
for k, (x, y, w, h) in enumerate(iso):
    crop = src.crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
    sc = min((cell_w - 70) / crop.width, (cell_h - 10) / crop.height, 3.0)
    crop = crop.resize((int(crop.width * sc), int(crop.height * sc)), Image.LANCZOS)
    cx, cy = (k % cols) * cell_w, (k // cols) * cell_h
    mont.paste(crop, (cx + 60, cy + 5))
    d.text((cx + 5, cy + 55), f"i{k}", fill="green", font=font)
    d.rectangle([cx, cy, cx + cell_w - 1, cy + cell_h - 1], outline="lightgray")
mont.save(SCRATCH + r"\isoline_montage.png")
