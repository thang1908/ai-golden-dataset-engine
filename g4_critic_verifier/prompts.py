from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "critic_verifier_v3"

CAPTION_FEW_SHOTS = """
Caption style examples only — never copy their facts into the current image:
- A middle-aged man with short, straight black hair wears a solid blue short-sleeve shirt, beige knee-length shorts, and slippers in a casual style.
- A young adult female with an average build and long black hair in a bun wears a solid beige short-sleeve T-shirt and a long white skirt.
- An adult male with short straight black hair wears a solid black short-sleeve T-shirt, white shorts, and slippers. He holds a phone and wears eyeglasses.
""".strip()


def generator_prompt(feedback: list[dict[str, str]] | None = None) -> str:
    text = (
        "Analyze the visible person in the image and return only the requested JSON: one factual "
        "20–35 word English caption in 1–3 short sentences and all 21 attributes with exact schema "
        "codes. Include every clearly visible material descriptor: age/gender presentation, build, "
        "hair, upper/lower clothing, footwear, bags/accessories, and style. Do not settle for a "
        "vague person-only caption when those details are visible; omit uncertain claims instead. "
        "Do not describe the setting, floor, background, walking, standing, or other activity. "
        "Do not infer identity, relationships, location, intent, or hidden details."
    )
    examples = "\n\n" + CAPTION_FEW_SHOTS
    return text + examples if not feedback else text + examples + " Fix only these prior issues:\n" + json.dumps(feedback)


def critic_prompt(draft: dict[str, Any]) -> str:
    return (
        "Inspect this image and draft annotation. Return only concrete unsupported, missing, or "
        "taxonomy-invalid claims as issues. Also report a material caption omission when a clearly "
        "visible age/gender presentation, hair, clothing, footwear, bag, or accessory is present "
        "in the image but absent from the caption. Do not require uncertain details and do not score "
        "it.\nDraft:\n"
        + json.dumps(draft, ensure_ascii=False)
    )


def verifier_prompt(draft: dict[str, Any], issues: list[dict[str, str]]) -> str:
    return (
        "Independently verify the draft annotation against the image and critic issues. Return "
        "accept only if the caption and all determined attributes are visibly supported; "
        "otherwise reject with concrete reasons. Do not use scores.\nDraft:\n"
        + json.dumps(draft, ensure_ascii=False)
        + "\nIssues:\n"
        + json.dumps(issues, ensure_ascii=False)
    )
