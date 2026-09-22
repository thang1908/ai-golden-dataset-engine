from __future__ import annotations

from abc import ABC, abstractmethod

from golden_dataset_harness.schemas.annotation import GroundingResult, QualityJudgment
from golden_dataset_harness.schemas.taxonomy import AttributeValue, Taxonomy


class BaseVisionLanguageModel(ABC):
    """Abstract base class for Vision Language Models."""

    def __init__(
        self,
        api_base_url: str = "",
        model_name: str = "",
        api_key: str = "",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize the model configuration."""
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Get the identifier of the model."""
        pass

    @abstractmethod
    async def generate_caption(self, image: bytes, prompt: str | None = None) -> str:
        """Generate a single caption for the given image."""
        pass

    @abstractmethod
    async def extract_attributes(self, image: bytes, taxonomy: Taxonomy) -> dict[str, AttributeValue]:
        """Extract attributes from the image constrained to the provided taxonomy."""
        pass

    @abstractmethod
    async def verify_claim(self, image: bytes, claim: str) -> GroundingResult:
        """Verify a specific claim against the image."""
        pass

    @abstractmethod
    async def judge_quality(
        self, image: bytes, caption: str, attributes: dict
    ) -> QualityJudgment:
        """Judge the quality of a generated annotation."""
        pass
