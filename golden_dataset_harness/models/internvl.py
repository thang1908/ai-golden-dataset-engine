from __future__ import annotations

from golden_dataset_harness.models.qwen_vl import QwenVLModel


class InternVLModel(QwenVLModel):
    """InternVL Model integration via OpenAI-compatible API."""

    def _prepare_messages(self, image: bytes, text: str) -> list[dict]:
        """Format the image and text with InternVL specific system prompt."""
        messages = super()._prepare_messages(image, text)
        return [
            {
                "role": "system",
                "content": "You are InternVL, a highly capable vision-language model.",
            }
        ] + messages
