from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageOps, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = None


def image_preview(
    path: str | Path,
    max_size: tuple[int, int] = (1800, 1100),
) -> Image.Image | None:
    """Load a bounded RGB preview without letting a bad asset break the UI."""
    candidate = Path(path)
    if not candidate.is_file():
        return None

    try:
        if candidate.suffix.casefold() == ".pdf":
            with fitz.open(candidate) as document:
                if document.page_count == 0:
                    return None
                page = document.load_page(0)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        else:
            with Image.open(candidate) as source:
                source.seek(0)
                image = ImageOps.exif_transpose(source).convert("RGB")

        image.thumbnail(max_size)
        return image.copy()
    except (fitz.FileDataError, RuntimeError, OSError, ValueError, UnidentifiedImageError):
        return None
