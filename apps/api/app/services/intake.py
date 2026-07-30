from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import BinaryIO, Any

from fastapi import HTTPException

from app.config import settings
from geoarchive.object_storage import (
    ObjectStorage,
    archived_upload_record,
    write_upload_policy,
)


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
    size = 0
    try:
        with temporary.open("wb") as output:
            while chunk := source.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "report_archive_too_large")
                output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, target)
    storage = ObjectStorage(settings)
    if storage.enabled:
        try:
            write_upload_policy(target, storage.archive_upload(target))
        except Exception as error:
            target.unlink(missing_ok=True)
            raise HTTPException(
                502, f"object_storage_upload_failed:{type(error).__name__}"
            ) from error
    return target


def register_inbox() -> dict[str, Any]:
    pipeline_path = str(settings.pipeline_root)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    from batch_intake import intake

    summary_path = intake(settings.inbox_root, settings.staging_root, settings.runs_root)
    for archive in settings.inbox_root.iterdir():
        if not archive.is_file() or archive.name.endswith(".policy.json"):
            continue
        policy = archive.with_suffix(archive.suffix + ".policy.json")
        marker = settings.staging_root / archive.stem / ".extracted.json"
        if marker.exists() and archived_upload_record(archive):
            archive.unlink(missing_ok=True)
            policy.unlink(missing_ok=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))
