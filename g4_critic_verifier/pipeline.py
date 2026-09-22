from __future__ import annotations

from typing import Any

from .client import VllmClient
from .errors import ResponseValidationError
from .prompts import critic_prompt, generator_prompt, verifier_prompt
from .taxonomy import attribute_json_schema, validate_attributes
from .translation import translate_caption

ANNOTATION_SCHEMA = {"type": "object", "properties": {"caption": {"type": "string", "minLength": 1, "maxLength": 500}, "attributes": attribute_json_schema()}, "required": ["caption", "attributes"], "additionalProperties": False}
CRITIC_SCHEMA = {"type": "object", "properties": {"issues": {"type": "array", "maxItems": 30, "items": {"type": "object", "properties": {"target": {"type": "string", "maxLength": 100}, "issue": {"type": "string", "minLength": 1, "maxLength": 500}, "suggested_correction": {"type": "string", "minLength": 1, "maxLength": 500}}, "required": ["target", "issue", "suggested_correction"], "additionalProperties": False}}}, "required": ["issues"], "additionalProperties": False}
VERIFIER_SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["accept", "reject"]}, "reasons": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 500}, "maxItems": 30}}, "required": ["decision", "reasons"], "additionalProperties": False}


def _annotation(value: dict[str, Any]) -> dict[str, Any]:
    try:
        caption = value["caption"]
        if set(value) != {"caption", "attributes"} or not isinstance(caption, str):
            raise ValueError
        caption = caption.strip()
        if not caption or len(caption) > 500:
            raise ValueError
        return {"caption": caption, "attributes": validate_attributes(value["attributes"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseValidationError("Model returned an invalid draft annotation") from exc


def _issues(value: dict[str, Any]) -> list[dict[str, str]]:
    issues = value.get("issues")
    if set(value) != {"issues"} or not isinstance(issues, list):
        raise ResponseValidationError("Model returned invalid critic issues")
    result = []
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {"target", "issue", "suggested_correction"} or not all(isinstance(issue[key], str) and issue[key].strip() for key in issue):
            raise ResponseValidationError("Model returned invalid critic issues")
        result.append({key: issue[key].strip() for key in issue})
    return result


def run(image: bytes, client: VllmClient, *, max_attempts: int = 3) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    feedback: list[dict[str, str]] | None = None
    for _ in range(max_attempts):
        draft = _annotation(client.complete(prompt=generator_prompt(feedback), schema_name="draft_annotation", schema=ANNOTATION_SCHEMA, image=image))
        issues = _issues(client.complete(prompt=critic_prompt(draft), schema_name="critic_issues", schema=CRITIC_SCHEMA, image=image))
        verification = client.complete(prompt=verifier_prompt(draft, issues), schema_name="verification", schema=VERIFIER_SCHEMA, image=image)
        decision, reasons = verification.get("decision"), verification.get("reasons")
        if decision not in {"accept", "reject"} or not isinstance(reasons, list) or not all(isinstance(reason, str) and reason.strip() for reason in reasons):
            raise ResponseValidationError("Model returned an invalid verifier decision")
        if decision == "accept":
            return {**draft, "caption_vi": translate_caption(client, draft["caption"])}
        feedback = issues + [{"target": "verifier", "issue": reason.strip(), "suggested_correction": "Correct the unsupported claim."} for reason in reasons]
    raise ResponseValidationError("Verifier rejected all bounded generation attempts")
