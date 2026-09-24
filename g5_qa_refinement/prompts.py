from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "qa_refinement_v3"

CAPTION_FEW_SHOTS = """
Caption style examples only — never copy their facts into the current image:
- A middle-aged man with short, straight black hair wears a solid blue short-sleeve shirt, beige knee-length shorts, and slippers in a casual style.
- A young adult female with an average build and long black hair in a bun wears a solid beige short-sleeve T-shirt and a long white skirt.
- An adult male with short straight black hair wears a solid black short-sleeve T-shirt, white shorts, and slippers. He holds a phone and wears eyeglasses.
""".strip()


def draft_prompt(corrections: list[str] | None = None) -> str:
    base = (
        "Analyze the visible person in the image and return only the requested JSON: one factual "
        "20–35 word English caption in 1–3 short sentences and all 21 attributes using the exact JSON "
        "Schema codes. Include all clearly visible material descriptors in this order when available: "
        "age/gender presentation, body build, hair, upper clothing, lower clothing, footwear, bags "
        "or accessories, and style. Do not use a vague person-only caption when those details are "
        "visible. Omit uncertain claims rather than guessing. Do not describe the setting, floor, "
        "background, walking, standing, or other activity. Do not infer identity, relationships, "
        "location, intent, or hidden details. "
        "For every single-label attribute, return one allowed code or null when it cannot be "
        "determined. For every multi-label attribute, return every visible allowed code; use "
        "[] only when the relevant region is visible and none apply, or null when it cannot "
        "be determined. Use unknown and none only where the schema allows them."
    )
    examples = "\n\n" + CAPTION_FEW_SHOTS
    return base + examples if not corrections else base + examples + " Correct these prior issues:\n" + json.dumps(corrections)


def questions_prompt(draft: dict[str, Any]) -> str:
    return (
        "Generate 3 to 12 concise verification questions for material caption claims, uncertain "
        "attributes, and likely caption omissions. Cover high-value visible descriptors such as "
        "age/gender presentation, hair, clothing, footwear, and accessories when relevant. Each "
        "question must have a unique id. Every item must contain exactly these non-empty string "
        "keys: id, target, question. Do not return null values, explanations, markdown, or extra "
        "keys. Return only "
        "JSON.\nDraft:\n" + json.dumps(draft, ensure_ascii=False)
    )


def answers_prompt(questions: list[dict[str, str]]) -> str:
    return (
        "Answer every supplied verification question exactly once using only the image. Keep "
        "the same question_id values. Every item must contain exactly question_id, answer, evidence, "
        "and determinable. answer and evidence must be non-empty strings; determinable must be a JSON "
        "boolean. When uncertain, set determinable to false and write a non-empty explanation such "
        "as 'Not determinable from this image' in both answer and evidence. Do not return null values, "
        "markdown, or extra keys. Return only JSON; do not infer hidden details.\nQuestions:\n"
        + json.dumps(questions, ensure_ascii=False)
    )


def compare_prompt(draft: dict[str, Any], answers: list[dict[str, Any]]) -> str:
    return "Compare the draft against image-grounded answers. Accept only if every stated claim is supported and the caption includes clearly determined material descriptors from the answers. Otherwise request refinement with concrete corrections. Do not require uncertain details. Return only JSON and no score.\nDraft:\n" + json.dumps(draft, ensure_ascii=False) + "\nAnswers:\n" + json.dumps(answers, ensure_ascii=False)
