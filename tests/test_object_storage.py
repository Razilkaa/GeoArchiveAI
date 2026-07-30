from pathlib import Path

from geoarchive.object_storage import ObjectStorage, archived_upload_record, write_upload_policy
from geoarchive.settings import Settings


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], int] = {}

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict,
    ) -> None:
        assert ExtraArgs["ContentType"]
        self.objects[(bucket, key)] = Path(filename).stat().st_size

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        return {
            "ContentLength": self.objects[(Bucket, Key)],
            "ETag": '"test-etag"',
        }


def test_object_storage_archives_upload_and_writes_policy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("S3_BUCKET", "geoarchive-reports")
    monkeypatch.setenv("S3_ACCESS_KEY", "access")
    monkeypatch.setenv("S3_SECRET_KEY", "secret")
    settings = Settings.from_environment()
    storage = ObjectStorage(settings)
    fake = FakeS3Client()
    storage._client = fake
    archive = tmp_path / "report.zip"
    archive.write_bytes(b"report")

    record = storage.archive_upload(archive)
    write_upload_policy(archive, record)

    assert record["bucket"] == "geoarchive-reports"
    assert record["size_bytes"] == 6
    assert record["key"].startswith("source/uploads/")
    assert record["key"].endswith("/report.zip")
    assert archived_upload_record(archive) == record


def test_partial_s3_configuration_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_SECRET_KEY", raising=False)

    try:
        ObjectStorage(Settings.from_environment())
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("partial S3 configuration must fail")
