"""Confidence scoring and routing node.

Fuses judge, consensus, and grounding scores into a single confidence value
and routes the annotation to auto-accept or human review based on a threshold.
Also assembles the final ``AnnotationRecord``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from golden_dataset_harness.schemas.annotation import (
    AnnotationRecord,
    PersonAttributes,
    ReviewStatus,
)
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)


async def confidence_scoring_node(
    state: PipelineState,
    *,
    weights: dict[str, float],
    threshold: float,
    model_version: str = "unknown",
) -> dict[str, Any]:
    """Compute fused confidence and build the final AnnotationRecord.

    Confidence formula::

        confidence = w_judge * judge_score
                   + w_consensus * consensus_score
                   + w_grounding * grounding_score

    Args:
        state: Must contain ``consensus``, ``attributes``, ``grounding_score``,
            and ``judge_result``.
        weights: Keys ``judge``, ``consensus``, ``grounding`` with float values.
        threshold: Minimum confidence for auto-acceptance.
        model_version: ID of the model used (for provenance).

    Returns:
        Partial state with ``confidence``, ``review_status``, and ``annotation``.
    """
    # Extract scores from state
    consensus = state.get("consensus")
    judge_result = state.get("judge_result")
    grounding_score: float = state.get("grounding_score", 0.0)

    judge_score = judge_result.quality_score if judge_result else 0.0
    consensus_score = consensus.agreement_score if consensus else 0.0

    # Weighted fusion
    confidence = (
        weights.get("judge", 0.4) * judge_score
        + weights.get("consensus", 0.3) * consensus_score
        + weights.get("grounding", 0.3) * grounding_score
    )
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    # Routing decision
    if confidence >= threshold:
        review_status = ReviewStatus.AUTO_ACCEPTED
    else:
        review_status = ReviewStatus.PENDING_REVIEW

    # Assemble the final record
    attributes = state.get("attributes", PersonAttributes())
    caption = consensus.caption if consensus else ""
    issues = judge_result.issues if judge_result else []
    image_id = state.get("image_id", "")
    image_path = state.get("image_path", "")

    annotation = AnnotationRecord(
        image_id=image_id,
        image_path=image_path,
        caption=caption,
        attributes=attributes,
        confidence=confidence,
        consensus_score=consensus_score,
        grounding_score=grounding_score,
        judge_score=judge_score,
        model_version=model_version,
        prompt_version="v1",
        review_status=review_status,
        issues=issues,
    )

    logger.info(
        "Confidence=%.3f → %s (judge=%.3f, consensus=%.3f, grounding=%.3f)",
        confidence,
        review_status.value,
        judge_score,
        consensus_score,
        grounding_score,
    )

    return {
        "confidence": confidence,
        "review_status": review_status.value,
        "annotation": annotation,
    }


def create_confidence_node(
    weights: dict[str, float] | None = None,
    threshold: float = 0.75,
    model_version: str = "unknown",
) -> Callable[[PipelineState], Any]:
    """Factory: bind scoring weights and threshold into a LangGraph node."""
    if weights is None:
        weights = {"judge": 0.4, "consensus": 0.3, "grounding": 0.3}

    async def node(state: PipelineState) -> dict[str, Any]:
        return await confidence_scoring_node(
            state, weights=weights, threshold=threshold, model_version=model_version
        )

    node.__name__ = "confidence_scoring"
    return node
