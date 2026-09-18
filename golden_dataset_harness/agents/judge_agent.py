"""Quality judge agent node.

Evaluates overall annotation quality by asking the VLM to assess
caption-image consistency, attribute correctness, hallucination risk,
and completeness.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import QualityJudgment
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)


async def quality_judge_node(
    state: PipelineState,
    *,
    vlm: BaseVisionLanguageModel,
) -> dict[str, Any]:
    """Evaluate the quality of the generated annotation.

    The judge assesses:
    1. Caption–image consistency
    2. Attribute correctness
    3. Hallucination detection
    4. Completeness (are important visible attributes captured?)

    Args:
        state: Must contain ``image_bytes``, ``consensus``, and ``attributes``.
        vlm: VLM used for quality evaluation.

    Returns:
        Partial state with ``judge_result`` (QualityJudgment).
    """
    image_bytes: bytes = state.get("image_bytes", b"")
    consensus = state.get("consensus")
    attributes = state.get("attributes")

    if not image_bytes:
        return {
            "judge_result": QualityJudgment(quality_score=0.0, issues=["No image"]),
            "errors": ["Judge: no image bytes"],
        }

    caption = consensus.caption if consensus else ""
    attr_dict = attributes.model_dump() if attributes else {}

    try:
        judgment = await vlm.judge_quality(image_bytes, caption, attr_dict)
    except Exception as e:
        logger.error("Quality judge failed: %s", e)
        judgment = QualityJudgment(
            quality_score=0.0,
            issues=[f"Judge error: {e}"],
        )

    logger.info(
        "Quality judge: score=%.3f, issues=%s",
        judgment.quality_score,
        judgment.issues,
    )
    return {"judge_result": judgment}


def create_judge_node(
    vlm: BaseVisionLanguageModel,
) -> Callable[[PipelineState], Any]:
    """Factory: bind VLM into a LangGraph-compatible judge node."""

    async def node(state: PipelineState) -> dict[str, Any]:
        return await quality_judge_node(state, vlm=vlm)

    node.__name__ = "quality_judge"
    return node
