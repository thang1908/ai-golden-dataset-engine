"""Safe image normalization for the API's Base64 message-size limit."""

from __future__ import annotations

import base64
import io

from PIL import Image, UnidentifiedImageError

from c1_vllm_medium.errors import ImageError

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
            return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"
        if quality > 50:
            quality -= 10
            continue
        width, height = source.size
        if max(width, height) <= 224:
            raise ImageError("Image cannot be compressed below the API content limit")
        source = source.resize(
            (max(1, int(width * 0.8)), max(1, int(height * 0.8))),
            Image.Resampling.LANCZOS,
        )
