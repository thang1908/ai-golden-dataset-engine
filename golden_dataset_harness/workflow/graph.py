"""LangGraph annotation pipeline — the core workflow orchestrator.

Builds a ``StateGraph`` that wires all agent nodes together with proper
fan-out (caption + attribute in parallel), fan-in (grounding waits for both),
and conditional routing (accept vs. review).

Usage::

    from golden_dataset_harness.workflow.graph import build_annotation_graph, load_settings

    settings = load_settings()
    graph = build_annotation_graph(settings)
    result = graph.invoke({"image_path": "person.jpg", "image_id": "001"})
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from golden_dataset_harness.agents.caption_agent import create_caption_node
from golden_dataset_harness.agents.attribute_agent import create_attribute_node
from golden_dataset_harness.agents.consensus import create_consensus_node
from golden_dataset_harness.agents.grounding_agent import create_grounding_node
from golden_dataset_harness.agents.judge_agent import create_judge_node
from golden_dataset_harness.agents.confidence import create_confidence_node
from golden_dataset_harness.models.factory import create_vlm
from golden_dataset_harness.schemas.annotation import ImageMetadata
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "configs" / "settings.yaml"


def load_settings(settings_path: str | None = None) -> dict[str, Any]:
    """Load pipeline settings from a YAML file.

    Args:
        settings_path: Path to settings YAML. Defaults to ``configs/settings.yaml``
            relative to the package root.

    Returns:
        Parsed settings dictionary.
    """
    path = Path(settings_path) if settings_path else _DEFAULT_SETTINGS_PATH
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Built-in nodes (load_image and terminal writers)
# ---------------------------------------------------------------------------

async def load_image_node(state: PipelineState) -> dict[str, Any]:
    """Read image from disk, extract metadata, and populate state."""
    image_path = state.get("image_path", "")
    image_bytes = state.get("image_bytes", b"")

    if not image_bytes and image_path:
        p = Path(image_path)
        if p.exists():
            image_bytes = p.read_bytes()
        else:
            logger.error("Image file not found: %s", image_path)
            return {"errors": [f"Image not found: {image_path}"]}

    if not image_bytes:
        return {"errors": ["No image data available"]}

    img = Image.open(io.BytesIO(image_bytes))
    metadata = ImageMetadata(
        image_id=state.get("image_id", Path(image_path).stem if image_path else "unknown"),
        image_path=image_path,
        width=img.width,
        height=img.height,
    )

    logger.info("Loaded image %s (%dx%d)", metadata.image_id, img.width, img.height)
    return {"image_bytes": image_bytes, "image_metadata": metadata}


async def write_accepted_node(state: PipelineState) -> dict[str, Any]:
    """Terminal node for auto-accepted annotations."""
    annotation = state.get("annotation")
    if annotation:
        logger.info("✅ Auto-accepted: %s (confidence=%.3f)", annotation.image_id, annotation.confidence)
    return {}


async def write_for_review_node(state: PipelineState) -> dict[str, Any]:
    """Terminal node for annotations routed to human review."""
    annotation = state.get("annotation")
    if annotation:
        logger.info(
            "🔍 Routed to review: %s (confidence=%.3f, issues=%s)",
            annotation.image_id,
            annotation.confidence,
            annotation.issues,
        )
    return {}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_confidence(state: PipelineState) -> str:
    """Conditional edge: route based on review_status."""
    status = state.get("review_status", "pending_review")
    if status == "auto_accepted":
        return "write_accepted"
    return "write_for_review"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_annotation_graph(settings: dict[str, Any]) -> CompiledStateGraph:
    """Build and compile the annotation pipeline graph.

    The graph topology::

        START
          │
        load_image
          ├──────────────────────┐
        caption_generation   attribute_extraction
          │                      │
        consensus                │
          ├──────────────────────┘
        grounding_verification
          │
        quality_judge
          │
        confidence_scoring
          ├─── (auto_accept) ──→ write_accepted ──→ END
          └─── (review)      ──→ write_for_review ──→ END

    Args:
        settings: Pipeline configuration dict (from ``load_settings``).

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()`` or
        ``.ainvoke()``.
    """
    # --- Create the VLM instance ---
    model_cfg = settings.get("model", {})
    vlm = create_vlm(
        provider=model_cfg.get("provider", "openai-compatible"),
    )

    # --- Create agent nodes via factories ---
    caption_cfg = settings.get("caption", {})
    consensus_cfg = settings.get("consensus", {})
    confidence_cfg = settings.get("confidence", {})

    taxonomy_path = str(Path(__file__).resolve().parent.parent / "configs" / "attributes.yaml")

    caption_node = create_caption_node(vlm, caption_cfg)
    attribute_node = create_attribute_node(vlm, taxonomy_path)
    consensus_node = create_consensus_node(
        embedding_model_name=consensus_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        similarity_threshold=consensus_cfg.get("similarity_threshold", 0.75),
    )
    grounding_node = create_grounding_node(vlm)
    judge_node = create_judge_node(vlm)
    confidence_node = create_confidence_node(
        weights={
            "judge": confidence_cfg.get("judge_weight", 0.4),
            "consensus": confidence_cfg.get("consensus_weight", 0.3),
            "grounding": confidence_cfg.get("grounding_weight", 0.3),
        },
        threshold=confidence_cfg.get("acceptance_threshold", 0.75),
        model_version=vlm.model_id,
    )

    # --- Build the StateGraph ---
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("load_image", load_image_node)
    builder.add_node("caption_generation", caption_node)
    builder.add_node("attribute_extraction", attribute_node)
    builder.add_node("consensus", consensus_node)
    builder.add_node("grounding_verification", grounding_node)
    builder.add_node("quality_judge", judge_node)
    builder.add_node("confidence_scoring", confidence_node)
    builder.add_node("write_accepted", write_accepted_node)
    builder.add_node("write_for_review", write_for_review_node)

    # Edges: START → load_image
    builder.add_edge(START, "load_image")

    # Fan-out: load_image → caption_generation AND attribute_extraction
    builder.add_edge("load_image", "caption_generation")
    builder.add_edge("load_image", "attribute_extraction")

    # caption_generation → consensus
    builder.add_edge("caption_generation", "consensus")

    # Fan-in: consensus AND attribute_extraction → grounding_verification
    builder.add_edge("consensus", "grounding_verification")
    builder.add_edge("attribute_extraction", "grounding_verification")

    # Sequential: grounding → judge → confidence
    builder.add_edge("grounding_verification", "quality_judge")
    builder.add_edge("quality_judge", "confidence_scoring")

    # Conditional routing after confidence
    builder.add_conditional_edges(
        "confidence_scoring",
        _route_after_confidence,
        {
            "write_accepted": "write_accepted",
            "write_for_review": "write_for_review",
        },
    )

    # Terminal edges
    builder.add_edge("write_accepted", END)
    builder.add_edge("write_for_review", END)

    graph = builder.compile()
    logger.info("Annotation graph compiled (provider=%s)", model_cfg.get("provider"))
    return graph
