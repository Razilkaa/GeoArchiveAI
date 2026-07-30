from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from pipeline.factory_runner import _api_image_payload


def test_api_image_payload_resizes_oversized_page(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    Image.new("RGB", (200, 100), "white").save(source)

    upload_name, payload = _api_image_payload(source, max_pixels=5_000)

    assert upload_name == "large_ocr.jpg"
    assert payload is not None
    with Image.open(io.BytesIO(payload)) as resized:
        assert resized.width * resized.height <= 5_000


def test_api_image_payload_keeps_normal_page(tmp_path: Path) -> None:
    source = tmp_path / "normal.jpg"
    Image.new("RGB", (20, 10), "white").save(source)

    upload_name, payload = _api_image_payload(source, max_pixels=5_000)

    assert upload_name == source.name
    assert payload is None
