"""Mock Vision Language Model for testing and development.

Returns deterministic, plausible outputs without requiring any GPU or API.
Uses image content hash for reproducible pseudo-random selection.
"""

from __future__ import annotations

import asyncio
import hashlib

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import GroundingResult, QualityJudgment
from golden_dataset_harness.schemas.taxonomy import AttributeValue, Taxonomy


# ---------------------------------------------------------------------------
# Caption pools — varied styles and content for realistic mock output
# ---------------------------------------------------------------------------

_CAPTION_POOL = [
    "A person wearing a black jacket and blue jeans walking on the sidewalk.",
    "A woman in a red dress carrying a handbag near a building entrance.",
    "A man in a white t-shirt and dark shorts with a backpack.",
    "A person with long hair wearing a green sweater and brown pants.",
    "A young woman in a gray hoodie and black leggings holding a phone.",
    "An elderly man wearing a beige coat and a baseball cap walking slowly.",
    "A teenager in a blue jacket carrying a shoulder bag across the street.",
    "A woman with short hair wearing a pink top and a dark skirt.",
]



class MockVLM(BaseVisionLanguageModel):
    """Mock Vision Language Model for testing and development.

    Produces deterministic outputs keyed on the MD5 hash of the input image,
    so the same image always yields the same results — useful for snapshot tests.
    """

    @property
    def model_id(self) -> str:
        return self.model_name or "mock-vlm-v1"

    def _hash_int(self, image: bytes) -> int:
        """Stable integer derived from image content."""
        return int(hashlib.md5(image).hexdigest(), 16)

    async def generate_caption(self, image: bytes, prompt: str | None = None) -> str:
        """Return a caption selected pseudo-randomly from the pool."""
        await asyncio.sleep(0.05)  # simulate latency
        h = self._hash_int(image)
        # Mix in the prompt text so different prompts yield different captions
        if prompt:
            h ^= hash(prompt)
        return _CAPTION_POOL[abs(h) % len(_CAPTION_POOL)]

    async def extract_attributes(
        self, image: bytes, taxonomy: Taxonomy
    ) -> dict[str, AttributeValue]:
        """Return attributes drawn from taxonomy values deterministically."""
        await asyncio.sleep(0.05)
        h = self._hash_int(image)
        result: dict[str, AttributeValue] = {}
        for index, (key, definition) in enumerate(taxonomy.items()):
            allowed = definition["classes"]
            selected = allowed[(h + index) % len(allowed)]
            if definition["type"] == "multi_label":
                result[key] = [selected] if (h + index) % 2 else []
            else:
                result[key] = selected
        return result

    async def verify_claim(self, image: bytes, claim: str) -> GroundingResult:
        """Return supported=False for known-bad claims, True otherwise."""
        await asyncio.sleep(0.05)
        lower = claim.lower()
        # Simulate hallucination detection for certain terms
        if any(term in lower for term in ("red hat", "umbrella", "motorcycle", "dog")):
            return GroundingResult(
                claim=claim,
                supported=False,
                confidence=0.92,
                reason="Object or attribute is not visible in the image.",
            )
        return GroundingResult(
            claim=claim,
            supported=True,
            confidence=0.87,
            reason="The claim is consistent with the image content.",
        )

    async def judge_quality(
        self, image: bytes, caption: str, attributes: dict
    ) -> QualityJudgment:
        """Return a plausible quality score."""
        await asyncio.sleep(0.05)
        # Slightly vary score based on caption length (longer = slightly better)
        base_score = 0.85
        bonus = min(len(caption) / 500, 0.10)
        return QualityJudgment(
            quality_score=round(base_score + bonus, 3),
            issues=[],
        )
