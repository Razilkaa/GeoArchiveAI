"""Parallel VLM reading of ALL manifest crops (context crops, hardened prompt).
Usage: set OPENAI_* env; python read_all_parallel.py <sheet_dir> [--model m] [--workers n]"""
import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

PROMPT = """Ты читаешь вырезку со скана советской структурной карты (сейсморазведка).
На вырезке рукописные чертёжные подписи. Центральная — главная, по краям соседние.
ПРАВИЛА:
1. Отметки глубин у пикетов: число с ДВУМЯ цифрами после точки (4.15, 3.85).
2. Подписи изогипс: КРУПНЫЙ шрифт, с минусом, одна-две цифры после точки (-3.8, -4.00).
3. Минус указывай ТОЛЬКО если штрих явно виден. Не догадывайся.
4. Обрезано/нечитаемо - не выдумывай, верни unreadable.
5. Прочитай ВСЕ различимые полные значения, каждое отдельным элементом.
Верни СТРОГО JSON:
{"values": [{"text": "...", "kind": "picket|isoline|profile_id|well|other", "confidence": 0..1}], "unreadable": false}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_dir", type=Path)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    sd = args.sheet_dir

    man = [json.loads(l) for l in (sd / "manifest_v4.jsonl").read_text(encoding="utf-8").splitlines()]
    out_path = sd / "readings.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines()}
    todo = [m for m in man if m["id"] not in done]
    print(f"total={len(man)} todo={len(todo)} model={args.model} workers={args.workers}")

    client = OpenAI()

    def read_one(m):
        img = (sd / m["context_crop"]).read_bytes()
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.model, temperature=0,
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": [
                              {"type": "image_url", "image_url": {
                                  "url": "data:image/png;base64," + base64.b64encode(img).decode()}},
                              {"type": "text", "text": "Прочитай вырезку. Только JSON."}]}])
            raw = resp.choices[0].message.content
            jm = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(jm.group(0)) if jm else {"values": [], "unreadable": True, "error": "no_json"}
        except Exception as e:  # noqa: BLE001
            parsed = {"values": [], "unreadable": True, "error": str(e)[:200]}
        return {"id": m["id"], "quad": m["quad"], "angle": m["angle"],
                "font_band": m["font_band"], "zone": m["zone"], **parsed,
                "latency_s": round(time.perf_counter() - t0, 2), "model": args.model}

    t0 = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(read_one, m) for m in todo]
        for k, fut in enumerate(as_completed(futs)):
            out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            out.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(todo)}")
    print(f"done {len(todo)} in {time.perf_counter() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
