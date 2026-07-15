"""Feasibility probe: can a VLM trace isoline polylines on a small tile,
verified against the raster (proposed points must lie on dark pixels)?"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

SRC = Path(r"C:\FINAM\Conference\384092\Графика\Tом 2\21.jpg")
OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
MODEL = os.environ.get("VLM_MODEL", "gpt-4o-mini")

# tile with clean isolines (bottom-central part of the survey)
X0, Y0, X1, Y1 = 2600, 4900, 3700, 5700
DOWN = 2  # model sees the tile downscaled 2x

PROMPT = """На изображении фрагмент структурной карты: плавные кривые линии (изогипсы),
прямые линии с засечками (сейсмопрофили), подписи цифр.

Задача: трассируй ИЗОГИПСЫ (только плавные кривые, НЕ прямые профили).
Для каждой видимой изогипсы верни последовательность точек вдоль неё (10-40 точек,
шаг примерно равномерный), в пиксельных координатах этого изображения.

Верни СТРОГО JSON:
{"polylines": [{"points": [[x1,y1],[x2,y2],...], "label_visible": "-3.2 или null"}]}
Не выдумывай линии: если кривая уходит за край, просто закончи точки у края."""


def main():
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    tile = img[Y0:Y1, X0:X1]
    small = cv2.resize(tile, (tile.shape[1] // DOWN, tile.shape[0] // DOWN))
    ok, buf = cv2.imencode(".png", small)
    b64 = base64.b64encode(buf).decode()

    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    raw = resp.choices[0].message.content
    jm = re.search(r"\{.*\}", raw, re.S)
    if not jm:
        sys.exit("no JSON in response: " + raw[:300])
    data = json.loads(jm.group(0))
    polys = data.get("polylines", [])
    print("polylines returned:", len(polys))

    # verification: fraction of points lying on (near) dark pixels
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.dilate(bw, np.ones((5, 5), np.uint8))  # 2px tolerance
    vis = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    total = on = 0
    for pl in polys:
        pts = np.array(pl["points"], dtype=int)
        pts[:, 0] = np.clip(pts[:, 0], 0, small.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, small.shape[0] - 1)
        hits = dark[pts[:, 1], pts[:, 0]] > 0
        total += len(pts)
        on += int(hits.sum())
        for (px, py), h in zip(pts, hits):
            cv2.circle(vis, (px, py), 3, (0, 180, 0) if h else (0, 0, 255), -1)
        cv2.polylines(vis, [pts], False, (255, 0, 0), 1)
        if pl.get("label_visible"):
            cv2.putText(vis, str(pl["label_visible"]), tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    print(f"points on raster lines: {on}/{total} = {on / max(1, total):.0%}")
    cv2.imencode(".png", vis)[1].tofile(str(OUT / "trace_tile_test.png"))
    (OUT / "trace_tile_test.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print("saved trace_tile_test.png / .json")


if __name__ == "__main__":
    main()
