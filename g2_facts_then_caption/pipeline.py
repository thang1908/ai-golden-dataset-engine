from __future__ import annotations

from typing import Any

from .agents.nodes import (
    caption_agent,
    structured_facts_agent,
    vietnamese_translation_agent,
    visual_analysis_agent,
)
from .client import VllmClient
from .telemetry import CallContext, CaseMetrics


def run(image: bytes, client: VllmClient, *, metrics: CaseMetrics | None = None, sample_id: str = "manual") -> dict[str, Any]:
    def context(stage: str) -> CallContext | None:
        return CallContext(sample_id, stage) if metrics else None
    analysis = visual_analysis_agent(image, client, metrics, context("visual_analysis"))
    facts, attributes = structured_facts_agent(image, analysis, client, metrics, context("structured_facts"))
    caption = caption_agent(facts, client, metrics, context("caption"))
    return {
        "caption": caption,
        "caption_vi": vietnamese_translation_agent(caption, client, metrics, context("caption_vietnamese")),
        "attributes": attributes,
    }
