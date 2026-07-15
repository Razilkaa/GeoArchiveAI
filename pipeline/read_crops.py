"""Read crops from manifest.jsonl with any OpenAI-compatible VLM endpoint
(vLLM serving GLM locally, or an external API key).

Usage:
  set OPENAI_BASE_URL=http://gpu-server:8000/v1   (or https://api.../v1)
  set OPENAI_API_KEY=...
  python read_crops.py output/21 --model glm-4.5v --limit 20

Writes output/<sheet>/readings.jsonl (resumable: already-read ids are skipped).
"""
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

PROMPT = (Path(__file__).parent / "output").glob("*/READING_PROMPT.md")


def load_prompt(sheet_dir: Path) -> str:
    text = (sheet_dir / "READING_PROMPT.md").read_text(encoding="utf-8")
    return text.split("## System / инструкция", 1)[1].split("## Пример ответа")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_dir", help="e.g. output/21")
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "glm-4.5v"))
    ap.add_argument("--limit", type=int, default=0, help="read only first N unread (0 = all)")
    args = ap.parse_args()

    sheet_dir = Path(__file__).parent / args.sheet_dir if not Path(args.sheet_dir).is_absolute() \
        else Path(args.sheet_dir)
    man_path = next(p for p in (sheet_dir / "manifest_v4.jsonl", sheet_dir / "manifest.jsonl") if p.exists())
    manifest = [json.loads(l) for l in man_path.read_text(encoding="utf-8").splitlines()]
    out_path = sheet_dir / "readings.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [m for m in manifest if m["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total={len(manifest)} done={len(done)} todo={len(todo)} model={args.model}")

    client = OpenAI()
    system = load_prompt(sheet_dir)

    import time
    with open(out_path, "a", encoding="utf-8") as out:
        for k, m in enumerate(todo):
            img_b64 = base64.b64encode((sheet_dir / m["crop"]).read_bytes()).decode()
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": "Прочитай вырезку. Только JSON."},
                        ]},
                    ],
                )
                raw = resp.choices[0].message.content
                jm = re.search(r"\{.*\}", raw, re.S)
                parsed = json.loads(jm.group(0)) if jm else {"values": [], "unreadable": True,
                                                             "error": "no_json", "raw": raw[:200]}
            except Exception as e:  # noqa: BLE001 - keep batch alive, record failure
                parsed = {"values": [], "unreadable": True, "error": str(e)[:200]}
            keep = {k: m[k] for k in ("bbox", "center", "band", "quad", "angle",
                                      "font_band", "zone", "detector_confidence") if k in m}
            rec = {"id": m["id"], **keep, **parsed,
                   "latency_s": round(time.perf_counter() - t0, 2), "model": args.model}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/{len(todo)}")
    print("done ->", out_path)


if __name__ == "__main__":
    main()
