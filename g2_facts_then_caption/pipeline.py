from __future__ import annotations

from typing import Any

from .agents.nodes import (
    caption_agent,
    structured_facts_agent,
    vietnamese_translation_agent,
    visual_analysis_agent,
)
from .client import VllmClient


def run(image: bytes, client: VllmClient) -> dict[str, Any]:
    analysis = visual_analysis_agent(image, client)
    facts, attributes = structured_facts_agent(image, analysis, client)
    caption = caption_agent(facts, client)
    return {
        "caption": caption,
        "caption_vi": vietnamese_translation_agent(caption, client),
        "attributes": attributes,
    }
