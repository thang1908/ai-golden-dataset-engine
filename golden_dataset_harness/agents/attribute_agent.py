"""Attribute extraction agent node.

Extracts structured person attributes from an image, constrained to an
externally-defined taxonomy loaded from YAML. Any value not in the taxonomy's
allowed list is replaced with ``"unknown"``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yaml

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import PersonAttributes
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)


def _load_taxonomy(path: str | Path) -> dict[str, list[str]]:
    """Load attribute taxonomy from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_against_taxonomy(
    raw: dict[str, str],
    taxonomy: dict[str, list[str]],
) -> dict[str, str]:
    """Replace any value not in the taxonomy's allowed list with 'unknown'."""
    validated: dict[str, str] = {}
    for attr, value in raw.items():
        allowed = taxonomy.get(attr)
        if allowed is not None and value not in allowed:
            logger.debug(
                "Attribute %r value %r not in taxonomy, resetting to 'unknown'",
                attr, value,
            )
            validated[attr] = "unknown"
        else:
            validated[attr] = value
    return validated


async def attribute_extraction_node(
    state: PipelineState,
    *,
    vlm: BaseVisionLanguageModel,
    taxonomy: dict[str, list[str]],
) -> dict[str, Any]:
    """Extract structured person attributes from the image.

    Args:
        state: Current pipeline state (must contain ``image_bytes``).
        vlm: Vision-language model for extraction.
        taxonomy: Mapping of attribute names to their allowed values.

    Returns:
        Partial state update with ``attributes`` (PersonAttributes).
    """
    image_bytes: bytes = state.get("image_bytes", b"")

    if not image_bytes:
        logger.error("No image bytes available for attribute extraction")
        return {
            "attributes": PersonAttributes(),
            "errors": ["Attribute extraction: no image bytes"],
        }

    raw_attributes = await vlm.extract_attributes(image_bytes, taxonomy)
    validated = _validate_against_taxonomy(raw_attributes, taxonomy)

    attributes = PersonAttributes(**validated)
    logger.info("Extracted attributes: %s", attributes.model_dump())
    return {"attributes": attributes}


def create_attribute_node(
    vlm: BaseVisionLanguageModel,
    taxonomy_path: str,
) -> Callable[[PipelineState], Any]:
    """Factory: load taxonomy once and bind VLM into a LangGraph node."""
    taxonomy = _load_taxonomy(taxonomy_path)

    async def node(state: PipelineState) -> dict[str, Any]:
        return await attribute_extraction_node(state, vlm=vlm, taxonomy=taxonomy)

    node.__name__ = "attribute_extraction"
    return node
