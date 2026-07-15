import csv
from PIL import Image, ImageDraw, ImageFont

SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad"
TILE = SCRATCH + r"\tile_center.png"
# (value, x, y) in tile pixel coords; tile origin = (2700, 2700) of full-res map
POINTS = [
    (2.97, 378, 88), (2.94, 340, 125), (2.54, 466, 219), (2.52, 466, 246),
    (2.70, 475, 297), (2.75, 445, 337), (2.68, 533, 322), (2.62, 585, 390),
    (2.67, 515, 400), (2.72, 468, 432), (2.77, 400, 407), (2.76, 380, 432),
    (3.02, 465, 546), (3.08, 508, 604), (2.79, 545, 560), (2.76, 560, 530),
    (2.85, 290, 540), (2.86, 300, 570), (3.10, 355, 663), (3.13, 420, 700),
    (3.17, 430, 745), (2.94, 610, 660), (2.85, 680, 645), (2.83, 790, 550),
    (2.84, 800, 590), (2.86, 760, 680), (2.68, 640, 500), (2.70, 720, 465),
    (2.69, 710, 545), (2.30, 750, 120), (2.36, 740, 165), (2.47, 955, 60),
    (2.51, 960, 90),
]
OX, OY = 2700, 2700

im = Image.open(TILE).convert("RGB")
d = ImageDraw.Draw(im)
try:
    font = ImageFont.truetype("arial.ttf", 22)
except OSError:
    font = ImageFont.load_default()

for v, x, y in POINTS:
    d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 0, 0), width=3)
    d.text((x + 12, y - 26), f"{v:.2f}", fill=(255, 0, 0), font=font)

im.save(SCRATCH + r"\tile_center_overlay.png")

with open(SCRATCH + r"\points_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["depth_km", "px_full", "py_full"])
    for v, x, y in POINTS:
        w.writerow([v, OX + x, OY + y])
print("ok", len(POINTS))
