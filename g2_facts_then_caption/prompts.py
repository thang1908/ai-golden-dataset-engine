from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "visual_facts_caption_v3"

CAPTION_FEW_SHOTS = """
Caption style examples only — never copy their facts into the current image:
- A middle-aged man with short, straight black hair wears a solid blue short-sleeve shirt, beige knee-length shorts, and slippers in a casual style.
- A young adult female with an average build and long black hair in a bun wears a solid beige short-sleeve T-shirt and a long white skirt.
- An adult male with short straight black hair wears a solid black short-sleeve T-shirt, white shorts, and slippers. He holds a phone and wears eyeglasses.
""".strip()


def visual_analysis_prompt() -> str:
    return (
        "Inspect the visible person in this image. Return only JSON with factual visible "
        "observations and uncertainties. Explicitly inventory caption-worthy evidence: age and "
        "gender presentation if clear, body build, hair length/texture/style/color, upper and "
        "lower clothing type/color/pattern, footwear, bags, accessories, carried objects, and "
        "style. Do not infer identity, relationships, location, intent, or occluded details."
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
        "Write a factual 20–35 word English caption in 1–3 short sentences using all material "
        "visible facts supplied below. The target is person-centric: omit the setting, floor, "
        "background, standing/walking, and other activity. Prefer this order: age/gender presentation when clearly "
        "supported, body build, hair, upper clothing, lower clothing, footwear, then bags or "
        "accessories. Do not collapse a detailed fact list into a vague 'a person wearing...' "
        "caption. Omit uncertain facts rather than guessing. Do not add identity, relationships, "
        "location, intent, or hidden details. "
        "Return only the requested JSON.\n\n"
        + CAPTION_FEW_SHOTS
        + "\nFacts:\n"
        + json.dumps(facts, ensure_ascii=False)
    )
