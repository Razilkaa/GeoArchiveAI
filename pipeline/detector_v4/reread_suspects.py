"""Re-read suspicious values with a hardened prompt on context crops, in parallel.

Suspects: value format inconsistent with font band (isoline-format in small font),
multi-value boxes, and 'other'-format values. Writes readings_reread.jsonl.
"""
import base64
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

OUT = Path(r"C:\FINAM\Conference\pipeline\output_v4\21")
MODEL = os.environ.get("VLM_MODEL", "gpt-4o-mini")
WORKERS = 12

PROMPT = """Ты читаешь вырезку со скана советской структурной карты (сейсморазведка, 1980 г.).
На вырезке рукописные чертёжные подписи. Центральная подпись - главная, по краям могут быть соседние.

ЖЁСТКИЕ ПРАВИЛА:
1. Отметки глубин у пикетов на ЭТОМ листе всегда имеют ДВЕ цифры после точки (например 2.72, 3.05).
   Если видишь только одну цифру после точки мелким шрифтом - вероятно, вторая цифра обрезана или слилась: перечитай внимательно.
2. Подписи изогипс написаны КРУПНЫМ шрифтом, одна цифра после точки, часто с минусом (-2.8).
3. Минус указывай ТОЛЬКО если штрих минуса явно виден. Не догадывайся.
4. Если цифры обрезаны краем вырезки или нечитаемы - НЕ выдумывай значение, верни unreadable для этого фрагмента.
5. Прочитай ВСЕ различимые полные значения, каждое отдельным элементом.

Верни СТРОГО JSON:
{"values": [{"text": "...", "kind": "picket|isoline|profile_id|well|other", "confidence": 0..1}], "unreadable": false}
Если ни одного надёжного значения нет: {"values": [], "unreadable": true}"""


def suspects():
    rows = list(csv.DictReader(open(OUT / "points_v4.csv", encoding="utf-8-sig")))
    ids = set()
    for r in rows:
        fmt_iso_small = r["kind"] == "isoline" and r["font_band"] == "small"
        multi = int(r["n_values_in_box"]) > 1
        other = r["kind"] == "other"
        if fmt_iso_small or multi or other:
            ids.add(int(r["box_id"]))
    return ids


def main():
    ids = suspects()
    man = {json.loads(l)["id"]: json.loads(l)
           for l in (OUT / "manifest_v4.jsonl").read_text(encoding="utf-8").splitlines()}
    out_path = OUT / "readings_reread.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines()}
    todo = [man[i] for i in sorted(ids) if i in man and i not in done]
    print(f"suspects={len(ids)} todo={len(todo)} model={MODEL} workers={WORKERS}")

    client = OpenAI()

    def read_one(m):
        img = (OUT / m["context_crop"]).read_bytes()
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(img).decode()}},
                        {"type": "text", "text": "Прочитай вырезку. Только JSON."},
                    ]},
                ],
            )
            raw = resp.choices[0].message.content
            jm = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(jm.group(0)) if jm else {"values": [], "unreadable": True, "error": "no_json"}
        except Exception as e:  # noqa: BLE001
            parsed = {"values": [], "unreadable": True, "error": str(e)[:200]}
        return {"id": m["id"], "quad": m["quad"], "angle": m["angle"],
                "font_band": m["font_band"], "zone": m["zone"], **parsed,
                "latency_s": round(time.perf_counter() - t0, 2), "model": MODEL,
                "pass": "reread_context"}

    t_start = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(read_one, m) for m in todo]
        for k, fut in enumerate(as_completed(futs)):
            out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            out.flush()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(todo)} elapsed={time.perf_counter() - t_start:.0f}s")
    print(f"done {len(todo)} in {time.perf_counter() - t_start:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
