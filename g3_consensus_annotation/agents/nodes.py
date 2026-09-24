"""G3 agent nodes."""

from __future__ import annotations

from typing import Any

from ..client import VllmClient
from ..prompts import caption_prompt, consensus_prompt, observer_prompt
from ..translation import translate_caption


def observer_agent(index: int, image: bytes, client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=observer_prompt(index), schema_name=f"observer_{index + 1}", schema=schema, image=image)


def consensus_agent(image: bytes, candidates: list[dict[str, Any]], client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=consensus_prompt(candidates), schema_name="attribute_consensus", schema=schema, image=image)


def caption_agent(attributes: dict[str, Any], client: VllmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return client.complete(prompt=caption_prompt(attributes), schema_name="caption", schema=schema)


def vietnamese_translation_agent(caption: str, client: VllmClient) -> str:
    return translate_caption(client, caption)

