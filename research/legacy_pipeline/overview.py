import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg"
OUT = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad\map21_overview.png"

im = Image.open(SRC)
ov = im.resize((im.width // 6, im.height // 6), Image.LANCZOS)
ov.save(OUT)
print(ov.size)
