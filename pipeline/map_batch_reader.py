from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps

from factory_runner import read_credentials


def read_manifest(path: Path, zone: str = "map_body") -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [record for record in records if record.get("zone") == zone]


def contact_sheet(
    manifest_path: Path,
    records: list[dict[str, Any]],
    destination: Path,
    columns: int = 5,
    cell_width: int = 320,
    cell_height: int = 230,
    crop_mode: str = "highlight",
) -> None:
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    root = manifest_path.parent
    for index, record in enumerate(records):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#888888", width=1)
        draw.rectangle((x, y, x + cell_width - 1, y + 32), fill="#e8f0fe")
        draw.text((x + 8, y + 3), f"ID {record['id']}", fill="black", font=font)
        crop_path = root / (record["crop"] if crop_mode == "tight" else record["context_crop"])
        with Image.open(crop_path) as crop:
            crop = crop.convert("RGB")
            if crop_mode.startswith("highlight"):
                x0, y0, width, height = record["bbox"]
                local_x = min(40, x0)
                local_y = min(40, y0)
                margin = 24 if crop_mode == "highlight_wide" else 0
                ImageDraw.Draw(crop).rectangle(
                    (
                        max(0, local_x - margin),
                        max(0, local_y - margin),
                        min(crop.width - 1, local_x + width + margin),
                        min(crop.height - 1, local_y + height + margin),
                    ),
                    outline="#ff0000",
                    width=4,
                )
            fitted = ImageOps.contain(crop, (cell_width - 12, cell_height - 42))
        px = x + (cell_width - fitted.width) // 2
        py = y + 36 + (cell_height - 40 - fitted.height) // 2
        sheet.paste(fitted, (px, py))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=90, optimize=True)


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Vision response is not a JSON object")
    return value


def read_sheet(client: OpenAI, model: str, sheet: Path, expected_ids: list[int]) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=5000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Это контактный лист пронумерованных кропов со старой геофизической карты. "
                            "Для каждого ID прочитай только подпись внутри красной рамки. Не объединяй соседние ячейки. "
                            "Верни JSON {\"items\":[{\"id\":число,\"text\":строка|null,"
                            "\"class\":\"numeric|profile|well|word|noise\",\"readable\":true|false}]}. "
                            "В items должен присутствовать каждый ID ровно один раз. Обрезки линий и засечки являются noise. "
                            f"Ожидаемые ID: {expected_ids}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url(sheet), "detail": "high"}},
                ],
            }
        ],
    )
    payload = parse_json(response.choices[0].message.content or "{}")
    payload["latency_s"] = round(time.perf_counter() - started, 2)
    payload["model"] = model
    return payload


def run(
    manifest_path: Path,
    credentials_file: Path,
    model: str,
    batch_size: int,
    workers: int,
    limit: int | None,
    external_approved: bool,
    crop_mode: str = "highlight",
) -> Path:
    if not external_approved:
        raise PermissionError("External processing is disabled. Pass --external-approved only for cleared maps.")
    records = read_manifest(manifest_path)
    if limit:
        records = records[:limit]
    output_dir = manifest_path.parent / f"batch_read_{crop_mode}"
    contacts_dir = output_dir / "contacts"
    batches = [records[index:index + batch_size] for index in range(0, len(records), batch_size)]
    pending = []
    for index, batch in enumerate(batches):
        result_path = output_dir / f"batch_{index:04d}.json"
        if result_path.exists():
            continue
        sheet_path = contacts_dir / f"batch_{index:04d}.jpg"
        contact_sheet(manifest_path, batch, sheet_path, crop_mode=crop_mode)
        pending.append((index, batch, sheet_path, result_path))

    creds = read_credentials(credentials_file)
    client = OpenAI(api_key=creds["OPENAI_API_KEY"], base_url=creds.get("BASE_URL"), timeout=180)
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(read_sheet, client, model, sheet, [int(item["id"]) for item in batch]): (index, result)
            for index, batch, sheet, result in pending
        }
        for future in as_completed(futures):
            index, result_path = futures[future]
            try:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(future.result(), ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as error:
                errors.append({"batch": index, "error": type(error).__name__, "detail": str(error)[:300]})
    if errors:
        (output_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Failed batches: {len(errors)}")

    by_id = {int(record["id"]): record for record in records}
    returned = {}
    latencies = []
    for result_path in sorted(output_dir.glob("batch_*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        latencies.append(float(payload.get("latency_s") or 0))
        for item in payload.get("items", []):
            item_id = int(item["id"])
            if item_id in by_id:
                returned[item_id] = item
    rows = []
    for item_id, record in by_id.items():
        reading = returned.get(item_id, {"id": item_id, "text": None, "class": "missing", "readable": False})
        rows.append({**record, "reading": reading})
    numeric_values = []
    for row in rows:
        reading = row["reading"]
        text = str(reading.get("text") or "").strip().replace(",", ".")
        if reading.get("class") == "numeric" and re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            reading["parsed_value"] = float(text)
            numeric_values.append(abs(float(text)))
    median_value = statistics.median(numeric_values) if numeric_values else 0.0
    outlier_limit = max(10.0, median_value * 5)
    for row in rows:
        reading = row["reading"]
        flags = []
        if reading.get("class") == "numeric" and "parsed_value" not in reading:
            flags.append("invalid_numeric")
        if abs(float(reading.get("parsed_value") or 0)) > outlier_limit:
            flags.append("numeric_outlier_or_profile")
        if reading.get("class") == "missing":
            flags.append("missing_response")
        reading["qc_flags"] = flags
        reading["requires_review"] = bool(flags)
    summary = {
        "manifest": str(manifest_path),
        "candidates": len(records),
        "requests": len(batches),
        "returned": len(returned),
        "readable": sum(bool(row["reading"].get("readable")) for row in rows),
        "classes": {},
        "qc_review": sum(bool(row["reading"]["requires_review"]) for row in rows),
        "numeric_median": round(median_value, 4),
        "numeric_outlier_limit": round(outlier_limit, 4),
        "sum_request_latency_s": round(sum(latencies), 2),
        "model": model,
    }
    for row in rows:
        name = str(row["reading"].get("class") or "missing")
        summary["classes"][name] = summary["classes"].get(name, 0) + 1
    output = output_dir / "results.json"
    output.write_text(json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Read many map crops with batched contact sheets")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--credentials-file", required=True, type=Path)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--crop-mode",
        choices=["highlight", "highlight_wide", "tight", "context"],
        default="highlight_wide",
    )
    parser.add_argument("--external-approved", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.manifest,
            args.credentials_file,
            args.model,
            args.batch_size,
            args.workers,
            args.limit,
            args.external_approved,
            args.crop_mode,
        )
    )


if __name__ == "__main__":
    main()
