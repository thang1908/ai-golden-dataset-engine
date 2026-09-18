"""LangGraph pipeline state definition.

The ``PipelineState`` TypedDict defines the complete state that flows through
the LangGraph StateGraph. Each node reads from and writes to this shared state.

Design note: We use ``Annotated`` with reducer functions where nodes may
append to a list (e.g., grounding results). For scalar fields that are
simply overwritten, no reducer is needed.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from golden_dataset_harness.schemas.annotation import (
    AnnotationRecord,
    CaptionCandidate,
    ConsensusResult,
    GroundingResult,
    ImageMetadata,
    PersonAttributes,
    QualityJudgment,
)


class PipelineState(TypedDict, total=False):
    """State schema for the annotation pipeline graph.

    Fields marked with ``Annotated[..., operator.add]`` use LangGraph's
    built-in list reducer — multiple nodes can append to the same list
    without overwriting each other.

    Fields *without* a reducer are overwritten by the last node to set them.
    """

    # --- Input ---
    image_id: str
    image_path: str
    image_bytes: bytes
    image_metadata: ImageMetadata

    # --- Caption Generation ---
    # Uses list reducer so the caption agent can append N candidates
    caption_candidates: Annotated[list[CaptionCandidate], operator.add]

    # --- Consensus ---
    consensus: ConsensusResult

    # --- Attribute Extraction ---
    attributes: PersonAttributes

    # --- Grounding ---
    # Uses list reducer so each claim verification appends its result
    grounding_results: Annotated[list[GroundingResult], operator.add]
    grounding_score: float

    # --- Quality Judge ---
    judge_result: QualityJudgment

    # --- Confidence & Routing ---
    confidence: float
    review_status: str

    # --- Final Output ---
    annotation: AnnotationRecord

    # --- Error handling ---
    errors: Annotated[list[str], operator.add]
