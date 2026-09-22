"""Grounding verification agent node.

Detects hallucinated attributes by converting the consensus caption and
extracted attributes into individual claims, then verifying each claim
against the source image using the VLM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import (
    GroundingResult,
    PersonAttributes,
)
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)


def _generate_claims(caption: str, attributes: PersonAttributes) -> list[str]:
    """Convert a caption and attributes into verifiable claims.

    Args:
        caption: The consensus caption text.
        attributes: Extracted person attributes.

    Returns:
        List of atomic claim strings.
    """
    claims: list[str] = []

    # The caption itself is a claim
    if caption:
        claims.append(caption)

    claim_templates = {
        "age": "The person's apparent age group is {value}.",
        "gender": "The person's apparent gender is {value}.",
        "body_build": "The person's visible body build is {value}.",
        "upper_clothing_type": "The person's upper clothing type is {value}.",
        "upper_clothing_color": "The person's upper clothing is {value}.",
        "lower_clothing_type": "The person's lower clothing type is {value}.",
        "lower_clothing_color": "The person's lower clothing is {value}.",
        "clothing_style": "The person's clothing style is {value}.",
        "upper_pattern": "The person's upper clothing pattern is {value}.",
        "footwear_type": "The person's footwear is {value}.",
        "bag_type": "The person has a {value}.",
        "bag_color": "A bag the person carries is {value}.",
        "headwear": "The person's headwear is {value}.",
        "eyewear": "The person's eyewear is {value}.",
        "face_mask": "The person's face covering is {value}.",
        "other_accessories": "The person wears a {value}.",
        "hair_length": "The person's hair length is {value}.",
        "hair_texture": "The person's hair texture is {value}.",
        "hairstyle": "The person's hairstyle is {value}.",
        "hair_color": "The person's hair color is {value}.",
        "carried_objects": "The person carries or accompanies a {value}.",
    }
    absence_claims = {
        "bag_type": "The person has no bag.",
        "bag_color": "The person has no bag with a color to annotate.",
        "other_accessories": "The person wears none of the listed other accessories.",
        "carried_objects": "The person carries or accompanies none of the listed objects.",
        "headwear": "The person wears no headwear.",
        "eyewear": "The person wears no eyewear.",
        "face_mask": "The person wears no face covering.",
    }
    for attr_name, value in attributes.model_dump().items():
        if value is None or value == "unknown":
            continue
        if value == [] or value == "none":
            claims.append(absence_claims[attr_name])
            continue
        values = value if isinstance(value, list) else [value]
        for code in values:
            if code != "unknown":
                claims.append(claim_templates[attr_name].format(value=code.replace("_", " ")))

    return claims


async def grounding_verification_node(
    state: PipelineState,
    *,
    vlm: BaseVisionLanguageModel,
) -> dict[str, Any]:
    """Verify generated claims against the source image.

    Args:
        state: Must contain ``image_bytes``, ``consensus``, and ``attributes``.
        vlm: VLM used for claim verification.

    Returns:
        Partial state with ``grounding_results`` and ``grounding_score``.
    """
    image_bytes: bytes = state.get("image_bytes", b"")
    consensus = state.get("consensus")
    attributes = state.get("attributes")

    if not image_bytes:
        return {
            "grounding_results": [],
            "grounding_score": 0.0,
            "errors": ["Grounding: no image bytes"],
        }

    caption = consensus.caption if consensus else ""
    attrs = attributes if attributes else PersonAttributes()

    claims = _generate_claims(caption, attrs)
    if not claims:
        return {"grounding_results": [], "grounding_score": 1.0}

    # Verify all claims concurrently
    tasks = [vlm.verify_claim(image_bytes, claim) for claim in claims]
    results: list[GroundingResult] = []
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, res in enumerate(raw_results):
        if isinstance(res, Exception):
            logger.warning("Grounding verification failed for claim %d: %s", i, res)
            results.append(
                GroundingResult(
                    claim=claims[i],
                    supported=False,
                    confidence=0.0,
                    reason=f"Verification error: {res}",
                )
            )
        else:
            results.append(res)

    # Aggregate score: mean confidence of supported claims, penalize unsupported
    if results:
        supported = [r for r in results if r.supported]
        unsupported = [r for r in results if not r.supported]
        if supported:
            score = (
                sum(r.confidence for r in supported) / len(results)
                - 0.1 * len(unsupported) / len(results)
            )
        else:
            score = 0.0
        grounding_score = max(0.0, min(1.0, score))
    else:
        grounding_score = 1.0

    logger.info(
        "Grounding: %d/%d claims supported, score=%.3f",
        sum(1 for r in results if r.supported),
        len(results),
        grounding_score,
    )
    return {"grounding_results": results, "grounding_score": grounding_score}


def create_grounding_node(
    vlm: BaseVisionLanguageModel,
) -> Callable[[PipelineState], Any]:
    """Factory: bind VLM into a LangGraph-compatible grounding node."""

    async def node(state: PipelineState) -> dict[str, Any]:
        return await grounding_verification_node(state, vlm=vlm)

    node.__name__ = "grounding_verification"
    return node
