from __future__ import annotations

from typing import Any

from ..client import VllmClient


def direct_annotation_agent(image: bytes, client: VllmClient) -> dict[str, Any]:
    """Generate the English caption and all 21 attributes in one vision call."""
    return client.annotate(image)


def vietnamese_translation_agent(caption: str, client: VllmClient) -> str:
    """Translate the final English caption after annotation is complete."""
    return client.translate_caption(caption)
