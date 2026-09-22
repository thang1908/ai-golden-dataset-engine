from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "critic_verifier_v1"


def generator_prompt(feedback: list[dict[str, str]] | None = None) -> str:
    text = (
        "Analyze the visible person in the image and return only the requested JSON: one "
        "concise factual English caption and all 21 attributes with exact schema codes. "
        "Do not infer identity, relationships, location, intent, or hidden details."
    )
    return text if not feedback else text + " Fix only these prior issues:\n" + json.dumps(feedback)


def critic_prompt(draft: dict[str, Any]) -> str:
    return (
        "Inspect this image and draft annotation. Return only concrete unsupported, missing, "
        "or taxonomy-invalid claims as issues. Do not score it.\nDraft:\n"
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

