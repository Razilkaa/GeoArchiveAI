from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import BinaryIO, Any

from app.config import settings


def safe_filename(name: str) -> str:
    filename = Path(name).name
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._() -]+", "_", filename).strip(" .")
    return cleaned or "archive.zip"


def save_upload(name: str, source: BinaryIO) -> Path:
    settings.inbox_root.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(name)
    target = settings.inbox_root / filename
    if target.exists():
        stem, suffix = target.stem, target.suffix
        index = 2
        while target.exists():
            target = settings.inbox_root / f"{stem}_{index}{suffix}"
            index += 1
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            output.write(chunk)
    os.replace(temporary, target)
    return target


def register_inbox() -> dict[str, Any]:
    pipeline_path = str(settings.pipeline_root)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    from batch_intake import intake

    summary_path = intake(settings.inbox_root, settings.staging_root, settings.runs_root)
    return json.loads(summary_path.read_text(encoding="utf-8"))
