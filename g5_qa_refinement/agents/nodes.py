"""G5 agent nodes."""

from __future__ import annotations

from typing import Any

from ..client import VllmClient
from ..prompts import answers_prompt, compare_prompt, draft_prompt, questions_prompt
from ..translation import translate_caption


def draft_agent(image: bytes, corrections: list[str] | None, client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=draft_prompt(corrections), schema_name="draft_annotation", schema=schema, image=image)


def question_agent(draft: dict[str, Any], client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=questions_prompt(draft), schema_name="verification_questions", schema=schema)


def answer_agent(image: bytes, questions: list[dict[str, str]], client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=answers_prompt(questions), schema_name="image_answers", schema=schema, image=image)


def compare_agent(draft: dict[str, Any], answers: list[dict[str, Any]], client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=compare_prompt(draft, answers), schema_name="comparison", schema=schema)


def vietnamese_translation_agent(caption: str, client: VllmClient) -> str:
    return translate_caption(client, caption)

