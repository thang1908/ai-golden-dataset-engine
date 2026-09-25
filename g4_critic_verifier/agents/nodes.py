"""G4 agent nodes."""

from __future__ import annotations

from typing import Any

from ..client import VllmClient
from ..prompts import critic_prompt, generator_prompt, verifier_prompt
from ..translation import translate_caption
from ..telemetry import CallContext


def generator_agent(image: bytes, feedback: list[dict[str, str]] | None, client: VllmClient, schema: dict[str, Any], context: CallContext) -> dict[str, Any]:
    return client.complete(prompt=generator_prompt(feedback), schema_name="draft_annotation", schema=schema, image=image, call_context=context)


def critic_agent(image: bytes, draft: dict[str, Any], client: VllmClient, schema: dict[str, Any], context: CallContext) -> dict[str, Any]:
    return client.complete(prompt=critic_prompt(draft), schema_name="critic_issues", schema=schema, image=image, call_context=context)


def verifier_agent(image: bytes, draft: dict[str, Any], issues: list[dict[str, str]], client: VllmClient, schema: dict[str, Any], context: CallContext) -> dict[str, Any]:
    return client.complete(prompt=verifier_prompt(draft, issues), schema_name="verification", schema=schema, image=image, call_context=context)


def vietnamese_translation_agent(caption: str, client: VllmClient, context: CallContext) -> str:
    return translate_caption(client, caption, context)
