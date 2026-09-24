from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "four_run_consensus_v3"

CAPTION_FEW_SHOTS = """
Caption style examples only — never copy their facts into the current image:
- A middle-aged man with short, straight black hair wears a solid blue short-sleeve shirt, beige knee-length shorts, and slippers in a casual style.
- A young adult female with an average build and long black hair in a bun wears a solid beige short-sleeve T-shirt and a long white skirt.
- An adult male with short straight black hair wears a solid black short-sleeve T-shirt, white shorts, and slippers. He holds a phone and wears eyeglasses.
""".strip()

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
        + " Write a factual medium-length English caption and all 21 attributes with exact schema "
        "codes. Include clear age/gender presentation, build, hair, outfit, footwear, and "
        "accessories rather than using a vague person-only caption. Omit uncertain claims. Do not "
        "infer identity, relationships, location, intent, or hidden details. "
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
        "Write a factual 20–35 word English caption based only on these final visible attribute "
        "codes. Keep it person-centric: no background, setting, walking, standing, or other activity. "
        "Include all determined high-value descriptors in this order when present: age and "
        "gender, body build, hair, upper clothing, lower clothing, footwear, then bags/accessories. "
        "Do not reduce the result to a vague person-only sentence and do not invent details not "
        "supported by the codes, identity, relationships, location, intent, or hidden details. "
        "Return only the requested JSON.\n\n"
        + CAPTION_FEW_SHOTS
        + "\nAttributes:\n"
        + json.dumps(attributes, ensure_ascii=False)
    )
