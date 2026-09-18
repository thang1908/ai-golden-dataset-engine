"""Caption generation agent node.

Generates N diverse caption candidates for a person image using varied prompt
templates. Uses asyncio.gather for concurrent generation across all N calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import CaptionCandidate
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)

# Prompt variants to encourage diversity in generated captions
_PROMPT_VARIANTS = [
    "Describe this person's appearance in one concise sentence.",
    "What does this person look like? Focus on clothing, accessories, and physical traits.",
    "Give a brief, factual description of the person in this image.",
    "Provide a detailed description of the individual, including clothing colors and style.",
    "Describe the person's outfit, hairstyle, and any items they are carrying.",
    "Write a single sentence summarizing this person's appearance for identification.",
    "What is this person wearing? Include colors, clothing type, and accessories.",
]


async def caption_generation_node(
    state: PipelineState,
    *,
    vlm: BaseVisionLanguageModel,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate multiple caption candidates for a single image.

    Args:
        state: Current pipeline state (must contain ``image_bytes``).
        vlm: Vision-language model to use for generation.
        config: Must contain ``num_candidates`` (int) and optionally ``temperature``.

    Returns:
        Partial state update with ``caption_candidates`` list.
    """
    num_candidates: int = config.get("num_candidates", 5)
    image_bytes: bytes = state.get("image_bytes", b"")

    if not image_bytes:
        logger.error("No image bytes available for caption generation")
        return {"caption_candidates": [], "errors": ["Caption generation: no image bytes"]}

    # Launch all generations concurrently
    tasks = [
        vlm.generate_caption(image_bytes, prompt=_PROMPT_VARIANTS[i % len(_PROMPT_VARIANTS)])
        for i in range(num_candidates)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[CaptionCandidate] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Caption candidate %d failed: %s", i, result)
            continue
        candidates.append(
            CaptionCandidate(
                text=result,
                model_id=vlm.model_id,
                prompt_version=f"v1-variant-{i % len(_PROMPT_VARIANTS)}",
            )
        )

    logger.info("Generated %d/%d caption candidates", len(candidates), num_candidates)
    return {"caption_candidates": candidates}


def create_caption_node(
    vlm: BaseVisionLanguageModel,
    config: dict[str, Any],
) -> Callable[[PipelineState], Any]:
    """Factory: bind VLM and config into a LangGraph-compatible node function."""

    async def node(state: PipelineState) -> dict[str, Any]:
        return await caption_generation_node(state, vlm=vlm, config=config)

    node.__name__ = "caption_generation"
    return node
