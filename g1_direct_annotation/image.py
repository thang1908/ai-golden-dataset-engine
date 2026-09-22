from __future__ import annotations

import base64
import io

from PIL import Image, UnidentifiedImageError

from .errors import ImageError

MAX_IMAGE_BYTES = 60_000


def to_data_url(image: bytes) -> str:
    try:
        source = Image.open(io.BytesIO(image)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("Input is not a supported image") from exc
    source.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    quality = 88
    while True:
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES:
            return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        if quality > 50:
            quality -= 10
        elif max(source.size) > 224:
            source = source.resize(
                tuple(max(1, int(v * 0.8)) for v in source.size), Image.Resampling.LANCZOS
            )
        else:
            raise ImageError("Image cannot be compressed below the API content limit")
