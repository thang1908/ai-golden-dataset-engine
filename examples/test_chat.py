"""Minimal smoke test for the remote vLLM Chat Completions API.

Run from the repository root:
    python examples/test_chat.py "Xin chào!"
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from golden_dataset_harness.models.oauth import OAuthTokenProvider
from golden_dataset_harness.models.provider_settings import ProviderSettings


async def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Xin chào!"
    settings = ProviderSettings()  # Reads golden_dataset_harness/.env

    async with httpx.AsyncClient(timeout=settings.vllm_timeout_seconds) as client:
        token = await OAuthTokenProvider(settings, client).get_token()
        headers = {"Authorization": f"Bearer {token}"}
        if prompt == "--list-models":
            response = await client.get(f"{settings.openai_base_url}/models", headers=headers)
            response.raise_for_status()
            for item in response.json().get("data", []):
                print(item["id"])
            return
        response = await client.post(
            f"{settings.openai_base_url}/chat/completions",
            headers=headers,
            json={
                "model": settings.vllm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if response.is_error:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"Chat API returned HTTP {response.status_code}: {detail}")
        print(response.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
