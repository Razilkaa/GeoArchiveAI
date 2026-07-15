import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg"

# args: name x0 y0 x1 y1 (full-res coords)
name, x0, y0, x1, y1 = sys.argv[1], *map(int, sys.argv[2:6])
im = Image.open(SRC)
crop = im.crop((x0, y0, x1, y1))
out = rf"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad\{name}.png"
crop.save(out)
print(out, crop.size)
