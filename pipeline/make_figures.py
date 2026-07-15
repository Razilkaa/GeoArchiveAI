"""Figures for the theses: fig1 = system architecture, fig2 = map fragment with detections."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = r"C:\FINAM\Conference\figures"
import os

os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fa", ec="#3a5a8c", fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, color="#3a5a8c"))


# level 1
box(0.5, 40, 23.5, 9, "Архивный отчёт\n(сканы: текст, графика)", fc="#f6efe2", ec="#8c6d3a", fs=8.2, bold=True)
box(27, 40, 22, 9, "OCR (PaddleOCR)\n+ LLM-корректор")
box(55, 40, 24, 9, "Агент-маршрутизатор:\nтипизация приложений\n(карты, схемы, разрезы)")
arrow(24.6, 44.5, 26.5, 44.5)
arrow(49.5, 44.5, 54.5, 44.5)

# level 2 branches
box(6, 22, 26, 10, "Текст отчёта:\nRAG-база знаний\n(гибридный поиск + reranker)")
box(38, 22, 26, 10, "Графика:\nCV-детекция объектов\n+ VLM-чтение кропов")
box(70, 22, 27, 10, "Структурированные данные:\nпикеты, изогипсы, профили,\nскважины, стратиграфия")
arrow(62, 39.5, 22, 32.7)
arrow(67, 39.5, 51, 32.7)
arrow(64.5, 27, 69.5, 27)

# level 3
box(22, 4, 30, 10, "QC-агент: перекрёстные проверки\nтекст ↔ карта, шаг изогипс,\nдоказательства (страница + кроп)")
box(60, 4, 27, 10, "Экспорт: GeoJSON /\nGeoPackage → QGIS, Petrel", fc="#e9f5ec", ec="#3a8c55", bold=True)
arrow(19, 21.5, 30, 14.7)
arrow(51, 21.5, 42, 14.7)
arrow(83.5, 21.5, 75, 14.7)
arrow(52.5, 9, 59.5, 9)

fig.savefig(OUT + r"\fig1_architecture.png", bbox_inches="tight", facecolor="white")
print("fig1 saved")

# fig2: fragment of the detection overlay
import cv2
import numpy as np

src = cv2.imdecode(np.fromfile(
    r"C:\FINAM\Conference\archive\map_experiments\digitization_test_overlay_v3.png", dtype=np.uint8), cv2.IMREAD_COLOR)
frag = src[80:680, 150:1050]
cv2.imencode(".png", frag)[1].tofile(OUT + r"\fig2_detection.png")
print("fig2 saved", frag.shape)
