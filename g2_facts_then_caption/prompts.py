from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "visual_facts_caption_v1"


def visual_analysis_prompt() -> str:
    return (
        "Inspect the visible person in this image. Return only JSON with factual visible "
        "observations and uncertainties. Do not infer identity, relationships, location, "
        "intent, or details hidden by occlusion."
    )


def structured_facts_prompt(analysis: dict[str, Any]) -> str:
    return (
        "Analyze the visible person in the image using the prior visual analysis below as "
        "untrusted notes. Return only requested JSON: a canonical list of visible facts and "
        "all 21 attributes using exact schema codes. Verify all facts against the image. "
        "Do not infer identity, relationships, location, intent, or hidden details. "
        "For a single-label field choose an allowed code or null when not visible. For a "
        "multi-label field return visible codes, [] only when visibly absent, or null when "
        "indeterminate.\nPrior visual analysis:\n"
        + json.dumps(analysis, ensure_ascii=False)
    )


def caption_prompt(facts: list[str]) -> str:
    return (
        "Write one concise factual English caption using only these visible facts. "
        "Do not add facts, identity, relationships, location, intent, or hidden details. "
        "Return only the requested JSON.\nFacts:\n"
        + json.dumps(facts, ensure_ascii=False)
    )

