"""Debug: what large text-like components exist on the tile?"""
import cv2
import numpy as np

SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d6c\scratchpad"
SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad"
img = cv2.imread(SCRATCH + r"\tile_center.png", cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)

print("components with 31<=h<=90:")
for i in range(1, n):
    x, y, w, h, a = stats[i]
    if 31 <= h <= 90 and w <= 200:
        asp = max(w, h) / max(1, min(w, h))
        fill = a / max(1, w * h)
        print(f"  ({x:4d},{y:4d}) w={w:3d} h={h:3d} area={a:5d} asp={asp:.1f} fill={fill:.2f}")
