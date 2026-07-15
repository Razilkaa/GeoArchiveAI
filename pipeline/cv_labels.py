"""CV pass: find label-like text boxes on the map tile, save numbered crops + montage."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad"
TILE = SCRATCH + r"\tile_center.png"

img = cv2.imread(TILE, cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# remove long lines: components that are very elongated / large
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
chars = np.zeros_like(bw)
for i in range(1, n):
    x, y, w, h, a = stats[i]
    if 6 <= h <= 45 and 2 <= w <= 45 and a >= 15 and max(w, h) / max(1, min(w, h)) < 6:
        chars[lab == i] = 255

# cluster chars into labels
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
blob = cv2.dilate(chars, kern)
cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if 30 <= w <= 160 and 18 <= h <= 160 and w * h < 15000:
        # require at least 3 char components inside
        m = chars[y:y + h, x:x + w]
        nn = cv2.connectedComponents(m)[0] - 1
        if nn >= 3:
            boxes.append((x, y, w, h))

boxes.sort(key=lambda b: (b[1] // 100, b[0]))
print("boxes:", len(boxes))

# overlay with box indices
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(color, str(i), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
cv2.imwrite(SCRATCH + r"\cv_boxes.png", color)

# montage of crops (3x upscaled) for reading
pad = 5
cell_w, cell_h = 260, 110
cols = 6
rows = (len(boxes) + cols - 1) // cols
mont = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
d = ImageDraw.Draw(mont)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()
src = Image.open(TILE)
for i, (x, y, w, h) in enumerate(boxes):
    crop = src.crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
    scale = min((cell_w - 60) / crop.width, (cell_h - 10) / crop.height, 3.5)
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
    mont.paste(crop, (cx + 55, cy + 5))
    d.text((cx + 5, cy + 40), f"#{i}", fill="red", font=font)
    d.rectangle([cx, cy, cx + cell_w - 1, cy + cell_h - 1], outline="lightgray")
mont.save(SCRATCH + r"\labels_montage.png")

np.save(SCRATCH + r"\boxes.npy", np.array(boxes))
print("saved montage", mont.size)
