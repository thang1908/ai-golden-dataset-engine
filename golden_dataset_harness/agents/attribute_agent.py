"""Extract and validate all SigLIP person attributes without losing multi-label values."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import PersonAttributes
from golden_dataset_harness.schemas.state import PipelineState
from golden_dataset_harness.schemas.taxonomy import Taxonomy, load_taxonomy, validate_attributes

logger = logging.getLogger(__name__)


_load_taxonomy = load_taxonomy
_validate_against_taxonomy = validate_attributes

async def attribute_extraction_node(
    state: PipelineState,
    *,
    vlm: BaseVisionLanguageModel,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Extract structured person attributes from the image.

    Args:
        state: Current pipeline state (must contain ``image_bytes``).
        vlm: Vision-language model for extraction.
        taxonomy: Attribute definitions including single/multi-label types and class codes.

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
