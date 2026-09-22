"""Smoke tests for the golden dataset harness pipeline.

These tests validate the full pipeline end-to-end using the MockVLM,
ensuring all components integrate correctly without requiring a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_IMAGE_PATH = Path(__file__).parent.parent / "images" / "astronaut.jpg"


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Load a real test image."""
    if SAMPLE_IMAGE_PATH.exists():
        return SAMPLE_IMAGE_PATH.read_bytes()
    # Fallback: generate a minimal valid JPEG
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 200), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_annotation_record_creation(self):
        from golden_dataset_harness.schemas.annotation import (
            AnnotationRecord,
            PersonAttributes,
            ReviewStatus,
        )
        record = AnnotationRecord(
            image_id="test_001",
            image_path="/tmp/test.jpg",
            caption="A person wearing a black jacket.",
            attributes=PersonAttributes(gender="male", upper_clothing_color="black"),
            confidence=0.85,
            consensus_score=0.9,
            grounding_score=0.8,
            judge_score=0.88,
            model_version="mock-vlm-v1",
        )
        assert record.image_id == "test_001"
        assert record.review_status == ReviewStatus.PENDING_REVIEW
        assert record.confidence == 0.85

    def test_score_clamping(self):
        from golden_dataset_harness.schemas.annotation import AnnotationRecord, PersonAttributes
        record = AnnotationRecord(
            image_id="test",
            image_path="",
            caption="test",
            attributes=PersonAttributes(),
            confidence=1.5,  # should be clamped to 1.0
            consensus_score=-0.1,  # should be clamped to 0.0
            grounding_score=0.5,
            judge_score=0.5,
            model_version="test",
        )
        assert record.confidence == 1.0
        assert record.consensus_score == 0.0

    def test_person_attributes_defaults(self):
        from golden_dataset_harness.schemas.annotation import PersonAttributes
        attrs = PersonAttributes()
        assert attrs.gender is None
        assert attrs.bag_type is None
        assert len(attrs.model_dump()) == 21


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestMockVLM:
    def test_create_mock(self):
        from golden_dataset_harness.models.factory import create_vlm
        vlm = create_vlm("mock")
        assert vlm.model_id == "mock-vlm-v1"

    def test_unknown_provider_raises(self):
        from golden_dataset_harness.models.factory import create_vlm
        with pytest.raises(ValueError, match="Unknown VLM provider"):
            create_vlm("nonexistent")

    @pytest.mark.asyncio
    async def test_generate_caption(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        vlm = create_vlm("mock")
        caption = await vlm.generate_caption(sample_image_bytes)
        assert isinstance(caption, str)
        assert len(caption) > 10

    @pytest.mark.asyncio
    async def test_extract_attributes(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        vlm = create_vlm("mock")
        from golden_dataset_harness.schemas.taxonomy import TAXONOMY
        taxonomy = TAXONOMY
        attrs = await vlm.extract_attributes(sample_image_bytes, taxonomy)
        assert "gender" in attrs
        assert attrs["gender"] in ["male", "female", "unknown"]

    @pytest.mark.asyncio
    async def test_verify_claim_supported(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        vlm = create_vlm("mock")
        result = await vlm.verify_claim(sample_image_bytes, "The person wears a jacket")
        assert result.supported is True
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_verify_claim_hallucinated(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        vlm = create_vlm("mock")
        result = await vlm.verify_claim(sample_image_bytes, "The person has a red hat")
        assert result.supported is False

    @pytest.mark.asyncio
    async def test_judge_quality(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        vlm = create_vlm("mock")
        judgment = await vlm.judge_quality(
            sample_image_bytes,
            "A person wearing a jacket.",
            {"gender": "male"},
        )
        assert judgment.quality_score > 0.5
        assert isinstance(judgment.issues, list)


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestAgents:
    @pytest.mark.asyncio
    async def test_caption_agent(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        from golden_dataset_harness.agents.caption_agent import caption_generation_node

        vlm = create_vlm("mock")
        state = {"image_bytes": sample_image_bytes, "image_id": "test"}
        result = await caption_generation_node(state, vlm=vlm, config={"num_candidates": 3})

        assert "caption_candidates" in result
        assert len(result["caption_candidates"]) == 3
        for c in result["caption_candidates"]:
            assert c.text
            assert c.model_id == "mock-vlm-v1"

    @pytest.mark.asyncio
    async def test_attribute_agent(self, sample_image_bytes):
        from golden_dataset_harness.models.factory import create_vlm
        from golden_dataset_harness.agents.attribute_agent import attribute_extraction_node

        vlm = create_vlm("mock")
        from golden_dataset_harness.schemas.taxonomy import TAXONOMY
        taxonomy = TAXONOMY
        state = {"image_bytes": sample_image_bytes}
        result = await attribute_extraction_node(state, vlm=vlm, taxonomy=taxonomy)

        assert "attributes" in result
        assert result["attributes"].gender in ["male", "female", "unknown"]

    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        from golden_dataset_harness.agents.confidence import confidence_scoring_node
        from golden_dataset_harness.schemas.annotation import (
            ConsensusResult, QualityJudgment, PersonAttributes,
        )

        state = {
            "image_id": "test",
            "image_path": "/tmp/test.jpg",
            "consensus": ConsensusResult(
                caption="A person wearing black.",
                agreement_score=0.9,
                all_candidates=[],
            ),
            "attributes": PersonAttributes(gender="male"),
            "grounding_score": 0.85,
            "judge_result": QualityJudgment(quality_score=0.88, issues=[]),
        }
        weights = {"judge": 0.4, "consensus": 0.3, "grounding": 0.3}
        result = await confidence_scoring_node(
            state, weights=weights, threshold=0.75, model_version="test"
        )

        assert "confidence" in result
        assert "annotation" in result
        expected = 0.4 * 0.88 + 0.3 * 0.9 + 0.3 * 0.85
        assert abs(result["confidence"] - expected) < 0.01


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_build_graph(self):
        """Test that the graph compiles without errors."""
        from golden_dataset_harness.workflow.graph import build_annotation_graph, load_settings
        settings = load_settings()
        settings["model"]["provider"] = "mock"
        graph = build_annotation_graph(settings)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_pipeline_end_to_end(self, sample_image_bytes):
        """Run the full pipeline on a test image using MockVLM."""
        from golden_dataset_harness.workflow.graph import build_annotation_graph, load_settings

        settings = load_settings()
        settings["model"]["provider"] = "mock"
        settings["caption"]["num_candidates"] = 1
        graph = build_annotation_graph(settings)

        result = await graph.ainvoke({
            "image_bytes": sample_image_bytes,
            "image_id": "smoke_test",
            "image_path": str(SAMPLE_IMAGE_PATH),
        })

        # Verify the pipeline produced an annotation
        assert result.get("annotation") is not None
        annotation = result["annotation"]
        assert annotation.image_id == "smoke_test"
        assert annotation.caption  # non-empty
        assert 0.0 <= annotation.confidence <= 1.0
        assert annotation.review_status in ("auto_accepted", "pending_review")
        from golden_dataset_harness.schemas.taxonomy import TAXONOMY, validate_attributes
        values = annotation.attributes.model_dump()
        assert list(values) == list(TAXONOMY)
        validate_attributes(values, require_all=True)
        assert all(v is not None for v in values.values())
        for key in ("bag_type", "bag_color", "other_accessories", "carried_objects"):
            assert isinstance(values[key], list)

        # Print for visibility
        print(f"\n{'='*60}")
        print(f"Caption: {annotation.caption}")
        print(f"Confidence: {annotation.confidence}")
        print(f"Status: {annotation.review_status}")
        print(f"Attributes: {annotation.attributes.model_dump()}")
        print(f"{'='*60}\n")
