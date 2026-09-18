from __future__ import annotations

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.models.internvl import InternVLModel
from golden_dataset_harness.models.mock_vlm import MockVLM
from golden_dataset_harness.models.qwen_vl import QwenVLModel


def create_vlm(provider: str, **kwargs) -> BaseVisionLanguageModel:
    """Create a Vision Language Model instance based on the specified provider.
    
    Args:
        provider: The provider name ('mock', 'qwen-vl', 'internvl').
        **kwargs: Additional configuration passed to the model constructor.
        
    Returns:
        An instance of a class derived from BaseVisionLanguageModel.
        
    Raises:
        ValueError: If an unknown provider is requested.
    """
    registry = {
        "mock": MockVLM,
        "qwen-vl": QwenVLModel,
        "internvl": InternVLModel,
    }

    if provider not in registry:
        raise ValueError(
            f"Unknown VLM provider: {provider}. Available providers: {list(registry.keys())}"
        )

    return registry[provider](**kwargs)
