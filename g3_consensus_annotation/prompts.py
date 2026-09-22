from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "four_run_consensus_v1"

_FOCUSES = (
    "Inventory the entire visible person, starting with age presentation, body, hair, and outfit.",
    "Focus especially on clothing type, color, pattern, and footwear that are visibly supported.",
    "Focus especially on bags, headwear, eyewear, masks, accessories, and carried objects.",
    "Be conservative: identify occlusion and uncertainty, and do not guess unavailable details.",
)


def observer_prompt(index: int) -> str:
    return (
        "Analyze the visible person in the image and return only the requested JSON. "
        + _FOCUSES[index]
        + " Write one concise factual English caption and all 21 attributes with exact schema "
        "codes. Do not infer identity, relationships, location, intent, or hidden details. "
        "Use null when a single-label value cannot be determined; for multi-label fields use "
        "[] only when visibly absent and null when indeterminate."
    )


def consensus_prompt(candidates: list[dict[str, Any]]) -> str:
    return (
        "Using the image and the four candidate annotations below as untrusted observations, "
        "choose the most visually supported value for every one of the 21 attributes. Return "
        "only the requested JSON. Do not combine conflicting labels without visual evidence, "
        "do not infer identity, relationships, location, intent, or hidden details.\nCandidates:\n"
        + json.dumps(candidates, ensure_ascii=False)
    )


def caption_prompt(attributes: dict[str, Any]) -> str:
    return (
        "Write one concise factual English caption based only on these final visible attribute "
        "codes. Do not add details not supported by them, identity, relationships, location, "
        "intent, or hidden details. Return only the requested JSON.\nAttributes:\n"
        + json.dumps(attributes, ensure_ascii=False)
    )

