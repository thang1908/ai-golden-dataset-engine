from __future__ import annotations

from typing import Any

from .client import VllmClient
from .agents.nodes import answer_agent, compare_agent, draft_agent, question_agent, vietnamese_translation_agent
from .errors import G5Error, ResponseValidationError
from .prompts import answers_prompt, compare_prompt, draft_prompt, questions_prompt
from .taxonomy import attribute_json_schema, validate_attributes
from .translation import translate_caption
from .telemetry import CallContext, SampleCallFactory

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


def _stage(stage: str, operation) -> dict[str, Any]:
    try:
        return operation()
    except G5Error as exc:
        raise ResponseValidationError(f"{stage}: {exc}") from exc


def _validated(stage: str, operation, validate):
    last_error: ResponseValidationError | None = None
    # The provider's schema mode can still emit malformed or locally invalid JSON.
    # Retry the model once after the transport-level retries in VllmClient are exhausted.
    for _ in range(2):
        try:
            return validate(_stage(stage, operation()))
        except ResponseValidationError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _questions(value: dict[str, Any]) -> list[dict[str, str]]:
    questions = value.get("questions")
    valid = (
        isinstance(questions, list)
        and questions
        and all(
            isinstance(item, dict)
            and set(item) == {"id", "target", "question"}
            and all(isinstance(item[key], str) and item[key].strip() for key in item)
            for item in questions
        )
        and len({item["id"] for item in questions}) == len(questions)
    )
    if not valid:
        raise ResponseValidationError("Model returned invalid verification questions")
    return questions


def _answers(value: dict[str, Any], question_ids: set[str]) -> list[dict[str, Any]]:
    answers = value.get("answers")
    valid = (
        isinstance(answers, list)
        and all(
            isinstance(item, dict)
            and set(item) == {"question_id", "answer", "evidence", "determinable"}
            and isinstance(item["question_id"], str)
            and isinstance(item["answer"], str)
            and item["answer"].strip()
            and isinstance(item["evidence"], str)
            and item["evidence"].strip()
            and isinstance(item["determinable"], bool)
            for item in answers
        )
        and {item["question_id"] for item in answers} == question_ids
    )
    if not valid:
        raise ResponseValidationError("Model returned incomplete image answers")
    return answers


def _comparison(value: dict[str, Any]) -> tuple[str, list[str]]:
    decision, corrections = value.get("decision"), value.get("corrections")
    if (
        decision not in {"accept", "refine"}
        or not isinstance(corrections, list)
        or not all(isinstance(item, str) and item.strip() for item in corrections)
    ):
        raise ResponseValidationError("Model returned invalid comparison")
    return decision, corrections


def run(image: bytes, client: VllmClient, *, max_refinements: int = 2, sample_id: str = "manual", run_id: str = "manual") -> dict[str, Any]:
    if max_refinements < 0: raise ValueError("max_refinements cannot be negative")
    corrections: list[str] | None = None
    calls = SampleCallFactory(run_id=run_id, method="g5", sample_id=sample_id)
    for round_index in range(max_refinements + 1):
        draft = _validated(
            "draft_annotation",
            lambda: lambda: draft_agent(image, corrections, client, ANNOTATION_SCHEMA, calls.next("draft_annotation")),
            _annotation,
        )
        questions = _validated(
            "verification_questions",
            lambda: lambda: question_agent(draft, client, QUESTION_SCHEMA, calls.next("verification_questions")),
            _questions,
        )
        question_ids = {item["id"] for item in questions}
        answers = _validated(
            "image_answers",
            lambda: lambda: answer_agent(image, questions, client, ANSWER_SCHEMA, calls.next("image_answers")),
            lambda value: _answers(value, question_ids),
        )
        decision, corrections = _validated(
            "comparison", lambda: lambda: compare_agent(draft, answers, client, COMPARE_SCHEMA, calls.next("comparison")), _comparison
        )
        if decision == "accept":
            caption_vi = _stage(
                "caption_vietnamese", lambda: vietnamese_translation_agent(draft["caption"], client, calls.next("caption_vietnamese"))
            )
            return {
                **draft,
                "caption_vi": caption_vi,
                "workflow_status": "accepted",
                "workflow_notes": [],
            }
        if round_index == max_refinements:
            caption_vi = _stage(
                "caption_vietnamese", lambda: vietnamese_translation_agent(draft["caption"], client, calls.next("caption_vietnamese"))
            )
            return {
                **draft,
                "caption_vi": caption_vi,
                "workflow_status": "refine_limit_reached",
                "workflow_notes": corrections,
            }
    raise ResponseValidationError("No draft was generated")
