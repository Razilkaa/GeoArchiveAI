from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from geoarchive.settings import Settings


class ObjectStorage:
    def __init__(self, settings: Settings):
        self.settings = settings
        configured = (
            settings.s3_endpoint_url,
            settings.s3_bucket,
            settings.s3_access_key,
            settings.s3_secret_key,
        )
        if any(configured) and not all(configured):
            raise ValueError("S3 configuration is incomplete")
        self.enabled = all(configured)
        self._client: Any | None = None

    @property
    def bucket(self) -> str:
        if not self.settings.s3_bucket:
            raise RuntimeError("S3 storage is disabled")
        return self.settings.s3_bucket

    @property
    def client(self) -> Any:
        if not self.enabled:
            raise RuntimeError("S3 storage is disabled")
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                region_name=self.settings.s3_region,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
        return self._client

    def ensure_bucket(self) -> None:
        if not self.enabled:
            return
        self.client.head_bucket(Bucket=self.bucket)

    def upload_file(self, path: Path, key: str) -> dict[str, Any]:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        return {
            "bucket": self.bucket,
            "key": key,
            "size_bytes": int(head["ContentLength"]),
            "etag": str(head.get("ETag", "")).strip('"'),
        }

    def archive_upload(self, path: Path) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        prefix = self.settings.s3_source_prefix or "source"
        key = (
            f"{prefix}/uploads/{now:%Y/%m/%d}/"
            f"{uuid4().hex}/{path.name}"
        )
        return self.upload_file(path, key)

    def upload_tree(self, root: Path, prefix: str) -> list[dict[str, Any]]:
        if not root.exists():
            return []
        uploaded = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            uploaded.append(self.upload_file(path, f"{prefix.rstrip('/')}/{relative}"))
        return uploaded


def write_upload_policy(archive: Path, record: dict[str, Any]) -> Path:
    policy = archive.with_suffix(archive.suffix + ".policy.json")
    temporary = policy.with_suffix(policy.suffix + ".part")
    payload = {
        "classification": "accessible_archive",
        "external_processing": True,
        "object_storage": record,
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(policy)
    return policy


def archived_upload_record(archive: Path) -> dict[str, Any] | None:
    policy = archive.with_suffix(archive.suffix + ".policy.json")
    if not policy.exists():
        return None
    try:
        payload = json.loads(policy.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    record = payload.get("object_storage")
    return record if isinstance(record, dict) and record.get("key") else None
