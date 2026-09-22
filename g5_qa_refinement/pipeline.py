from __future__ import annotations

from typing import Any

from .client import VllmClient
from .errors import ResponseValidationError
from .prompts import answers_prompt, compare_prompt, draft_prompt, questions_prompt
from .taxonomy import attribute_json_schema, validate_attributes
from .translation import translate_caption

ANNOTATION_SCHEMA = {"type": "object", "properties": {"caption": {"type": "string", "minLength": 1, "maxLength": 500}, "attributes": attribute_json_schema()}, "required": ["caption", "attributes"], "additionalProperties": False}
QUESTION_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "minItems": 1, "maxItems": 30, "items": {"type": "object", "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 50}, "target": {"type": "string", "minLength": 1, "maxLength": 100}, "question": {"type": "string", "minLength": 1, "maxLength": 400}}, "required": ["id", "target", "question"], "additionalProperties": False}}}, "required": ["questions"], "additionalProperties": False}
ANSWER_SCHEMA = {"type": "object", "properties": {"answers": {"type": "array", "maxItems": 30, "items": {"type": "object", "properties": {"question_id": {"type": "string", "minLength": 1}, "answer": {"type": "string", "minLength": 1, "maxLength": 400}, "evidence": {"type": "string", "minLength": 1, "maxLength": 400}, "determinable": {"type": "boolean"}}, "required": ["question_id", "answer", "evidence", "determinable"], "additionalProperties": False}}}, "required": ["answers"], "additionalProperties": False}
COMPARE_SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["accept", "refine"]}, "corrections": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 500}, "maxItems": 30}}, "required": ["decision", "corrections"], "additionalProperties": False}


def _annotation(value: dict[str, Any]) -> dict[str, Any]:
    try:
        caption = value["caption"]
        if set(value) != {"caption", "attributes"} or not isinstance(caption, str): raise ValueError
        caption = caption.strip()
        if not caption or len(caption) > 500: raise ValueError
        return {"caption": caption, "attributes": validate_attributes(value["attributes"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseValidationError("Model returned an invalid draft") from exc


def run(image: bytes, client: VllmClient, *, max_refinements: int = 2) -> dict[str, Any]:
    if max_refinements < 0: raise ValueError("max_refinements cannot be negative")
    corrections: list[str] | None = None
    for round_index in range(max_refinements + 1):
        draft = _annotation(client.complete(prompt=draft_prompt(corrections), schema_name="draft_annotation", schema=ANNOTATION_SCHEMA, image=image))
        question_value = client.complete(prompt=questions_prompt(draft), schema_name="verification_questions", schema=QUESTION_SCHEMA)
        questions = question_value.get("questions")
        if not isinstance(questions, list) or len({item.get("id") for item in questions if isinstance(item, dict)}) != len(questions): raise ResponseValidationError("Model returned invalid verification questions")
        answer_value = client.complete(prompt=answers_prompt(questions), schema_name="image_answers", schema=ANSWER_SCHEMA, image=image)
        answers = answer_value.get("answers")
        question_ids = {item["id"] for item in questions}
        if not isinstance(answers, list) or {item.get("question_id") for item in answers if isinstance(item, dict)} != question_ids: raise ResponseValidationError("Model returned incomplete image answers")
        comparison = client.complete(prompt=compare_prompt(draft, answers), schema_name="comparison", schema=COMPARE_SCHEMA)
        decision, corrections = comparison.get("decision"), comparison.get("corrections")
        if decision not in {"accept", "refine"} or not isinstance(corrections, list) or not all(isinstance(item, str) and item.strip() for item in corrections): raise ResponseValidationError("Model returned invalid comparison")
        if decision == "accept":
            return {**draft, "caption_vi": translate_caption(client, draft["caption"])}
        if round_index == max_refinements: break
    raise ResponseValidationError("Comparison requested refinement after the bounded limit")
