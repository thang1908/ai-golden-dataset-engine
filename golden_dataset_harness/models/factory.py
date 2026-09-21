from __future__ import annotations

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.models.mock_vlm import MockVLM
from golden_dataset_harness.models.openai_compatible import OpenAICompatibleVLM


def create_vlm(provider: str, **kwargs) -> BaseVisionLanguageModel:
    """Create a Vision Language Model instance based on the specified provider.

    Args:
        provider: The provider name (``openai-compatible`` or ``mock`` for tests).
        **kwargs: Additional configuration passed to the model constructor.

    Returns:
        An instance of a class derived from BaseVisionLanguageModel.

    Raises:
        ValueError: If an unknown provider is requested.
    """
    registry = {
        "mock": MockVLM,
        "openai-compatible": OpenAICompatibleVLM,
    }

    if provider not in registry:
        raise ValueError(
            f"Unknown VLM provider: {provider}. Available providers: {list(registry.keys())}"
        )

    return registry[provider](**kwargs)
