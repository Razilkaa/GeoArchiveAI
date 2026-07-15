"""Collect character-size statistics over the full map sheet."""
import cv2
import numpy as np

SRC = r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg"

img = cv2.imdecode(np.fromfile(SRC, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
print("map:", img.shape)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
h = stats[1:, cv2.CC_STAT_HEIGHT]
w = stats[1:, cv2.CC_STAT_WIDTH]
a = stats[1:, cv2.CC_STAT_AREA]

# text-like: not hairline, not huge, moderate aspect, reasonable fill
asp = np.maximum(w, h) / np.maximum(1, np.minimum(w, h))
fill = a / np.maximum(1, w * h)
m = (h >= 6) & (h <= 120) & (w >= 3) & (w <= 120) & (asp < 5) & (fill > 0.15) & (a > 15)
hh = h[m]
print("text-like components:", hh.size)

hist, edges = np.histogram(hh, bins=np.arange(5, 125, 5))
for c, e in zip(hist, edges):
    bar = "#" * int(60 * c / max(1, hist.max()))
    print(f"h {e:3d}-{e+4:3d}: {c:6d} {bar}")

for q in (5, 25, 50, 75, 90, 95, 99):
    print(f"p{q}: {np.percentile(hh, q):.0f}")
