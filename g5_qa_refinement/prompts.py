from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "qa_refinement_v1"


def draft_prompt(corrections: list[str] | None = None) -> str:
    base = "Analyze the visible person in the image and return only the requested JSON: a concise factual English caption and all 21 attributes. Do not infer identity, relationships, location, intent, or hidden details."
    return base if not corrections else base + " Correct these prior issues:\n" + json.dumps(corrections)


def questions_prompt(draft: dict[str, Any]) -> str:
    return "Generate concise verification questions for material caption claims and uncertain attributes in this draft. Return only JSON.\nDraft:\n" + json.dumps(draft, ensure_ascii=False)


def answers_prompt(questions: list[dict[str, str]]) -> str:
    return "Answer these verification questions only from the image. Return only JSON; do not infer hidden details.\nQuestions:\n" + json.dumps(questions, ensure_ascii=False)


def compare_prompt(draft: dict[str, Any], answers: list[dict[str, Any]]) -> str:
    return "Compare the draft against image-grounded answers. Accept only if supported; otherwise request refinement with concrete corrections. Return only JSON and no score.\nDraft:\n" + json.dumps(draft, ensure_ascii=False) + "\nAnswers:\n" + json.dumps(answers, ensure_ascii=False)

