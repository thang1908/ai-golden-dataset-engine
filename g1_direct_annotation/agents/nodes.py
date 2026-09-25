from __future__ import annotations

from typing import Any

from ..client import VllmClient
from ..telemetry import CallContext, CaseMetrics


def direct_annotation_agent(image: bytes, client: VllmClient, metrics: CaseMetrics | None = None, context: CallContext | None = None) -> dict[str, Any]:
    """Generate the English caption and all 21 attributes in one vision call."""
    return client.annotate(image, metrics, context)


def vietnamese_translation_agent(caption: str, client: VllmClient, metrics: CaseMetrics | None = None, context: CallContext | None = None) -> str:
    """Translate the final English caption after annotation is complete."""
    return client.translate_caption(caption, metrics, context)
