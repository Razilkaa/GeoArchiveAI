"""Two-band detector: piket labels (small font) vs isoline labels (large font)."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad"
TILE = SCRATCH + r"\tile_center.png"

img = cv2.imread(TILE, cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
chars = np.zeros_like(bw)
char_h = {}
for i in range(1, n):
    x, y, w, h, a = stats[i]
    asp = max(w, h) / max(1, min(w, h))
    fill = a / max(1, w * h)
    if 8 <= h <= 90 and 3 <= w <= 90 and a >= 15 and asp < 5 and fill > 0.15:
        chars[lab == i] = 255
        char_h[i] = h

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
blob = cv2.dilate(chars, kern)
cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []  # (x,y,w,h, cls)
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w < 25 or h < 15 or w > 400 or h > 250:
        continue
    m_lab = lab[y:y + h, x:x + w]
    hs = [char_h[i] for i in np.unique(m_lab) if i in char_h]
    if len(hs) < 2:
        continue
    med = float(np.median(hs))
    cls = "piket" if med <= 30 else "isoline"
    boxes.append((x, y, w, h, cls, med))

boxes.sort(key=lambda b: (b[1] // 100, b[0]))
np_boxes = np.array([(x, y, w, h, 0 if c == "piket" else 1) for x, y, w, h, c, _ in boxes])
np.save(SCRATCH + r"\boxes2.npy", np_boxes)
print("total:", len(boxes), " piket:", sum(1 for b in boxes if b[4] == "piket"),
      " isoline:", sum(1 for b in boxes if b[4] == "isoline"))

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
iso_idx = []
for i, (x, y, w, h, cls, med) in enumerate(boxes):
    col = (0, 0, 255) if cls == "piket" else (0, 180, 0)
    cv2.rectangle(color, (x, y), (x + w, y + h), col, 2 if cls == "piket" else 3)
    if cls == "isoline":
        cv2.putText(color, str(i), (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
        iso_idx.append(i)
cv2.imwrite(SCRATCH + r"\tile_overlay_v3.png", color)

# montage of isoline crops only
src = Image.open(TILE)
pad = 8
cell_w, cell_h = 300, 130
cols = 4
rows = (len(iso_idx) + cols - 1) // cols if iso_idx else 1
mont = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
d = ImageDraw.Draw(mont)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    font = ImageFont.load_default()
for k, i in enumerate(iso_idx):
    x, y, w, h, cls, med = boxes[i]
    crop = src.crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
    sc = min((cell_w - 70) / crop.width, (cell_h - 10) / crop.height, 3.0)
    crop = crop.resize((int(crop.width * sc), int(crop.height * sc)), Image.LANCZOS)
    cx, cy = (k % cols) * cell_w, (k // cols) * cell_h
    mont.paste(crop, (cx + 60, cy + 5))
    d.text((cx + 5, cy + 50), f"#{i}", fill="green", font=font)
    d.rectangle([cx, cy, cx + cell_w - 1, cy + cell_h - 1], outline="lightgray")
mont.save(SCRATCH + r"\isoline_montage.png")
print("isoline boxes:", iso_idx)
