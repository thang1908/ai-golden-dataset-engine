"""Contract tests for OAuth and the OpenAI-compatible vision adapter."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from golden_dataset_harness.models.oauth import OAuthError, OAuthTokenProvider
from golden_dataset_harness.models.openai_compatible import OpenAICompatibleVLM, VLMServiceError
from golden_dataset_harness.models.provider_settings import ProviderSettings


def make_settings(**overrides) -> ProviderSettings:
    values = {
        "vllm_base_url": "https://vlm.example.test",
        "vllm_client_id": "client-1",
        "vllm_client_secret": "secret-1",
        "vllm_project_id": "project-1",
        "vllm_model": "v-llm-v1-large",
        "vllm_max_retries": 2,
        "vllm_token_refresh_skew_seconds": 10,
    }
    values.update(overrides)
    return ProviderSettings(_env_file=None, **values)


def jpeg_bytes(size=(640, 480)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 40, 180)).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_oauth_token_is_cached_and_refreshed_before_expiry():
    requests: list[dict] = []
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "access_token": f"token-{len(requests)}", "expires_in": 100,
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OAuthTokenProvider(make_settings(), client, clock=lambda: now[0])
    assert await provider.get_token() == "token-1"
    assert await provider.get_token() == "token-1"
    now[0] = 191.0
    assert await provider.get_token() == "token-2"
    assert len(requests) == 2
    assert requests[0] == {
        "client_id": "client-1", "client_secret": "secret-1", "project_id": "project-1"
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_oauth_response_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={"token_type": "bearer"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OAuthTokenProvider(make_settings(), client)
    with pytest.raises(OAuthError, match="access_token"):
        await provider.get_token()
    await client.aclose()


@pytest.mark.asyncio
async def test_401_refreshes_token_and_structured_attributes_are_validated():
    auth_count = 0
    chat_requests: list[httpx.Request] = []

    def auth_handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_count
        auth_count += 1
        return httpx.Response(200, request=request, json={
            "access_token": f"token-{auth_count}", "expires_in": 3600,
        })

    def chat_handler(request: httpx.Request) -> httpx.Response:
        chat_requests.append(request)
        if len(chat_requests) == 1:
            return httpx.Response(401, request=request, json={"error": "expired"})
        return httpx.Response(200, request=request, json={
            "choices": [{"message": {"content": json.dumps({
                "gender": "female", "upper_color": "white",
            })}}]
        })

    auth_client = httpx.AsyncClient(transport=httpx.MockTransport(auth_handler))
    chat_client = httpx.AsyncClient(transport=httpx.MockTransport(chat_handler))
    settings = make_settings(vllm_guardrail="large", vllm_guard_output_mode="collect")
    provider = OAuthTokenProvider(settings, auth_client)
    model = OpenAICompatibleVLM(settings, provider, chat_client)
    result = await model.extract_attributes(
        jpeg_bytes(), {"gender": ["male", "female", "unknown"],
                       "upper_color": ["white", "black", "unknown"]}
    )
    assert result == {"gender": "female", "upper_color": "white"}
    assert auth_count == 2
    assert chat_requests[0].headers["authorization"] == "Bearer token-1"
    assert chat_requests[1].headers["authorization"] == "Bearer token-2"
    payload = json.loads(chat_requests[1].content)
    assert payload["model"] == "v-llm-v1-large"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["guardrail"] == "large"
    assert payload["guard_output_mode"] == "collect"
    assert "extra_body" not in payload
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert len(base64.b64decode(image_url.split(",", 1)[1])) <= 60_000
    await auth_client.aclose()
    await chat_client.aclose()


@pytest.mark.asyncio
async def test_attribute_value_outside_taxonomy_is_rejected():
    class TokenProvider:
        async def get_token(self, force_refresh=False):
            return "token"

        async def invalidate(self):
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={
            "choices": [{"message": {"content": '{"gender":"robot"}'}}]
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleVLM(make_settings(), TokenProvider(), client)
    with pytest.raises(VLMServiceError, match="gender"):
        await model.extract_attributes(jpeg_bytes(), {"gender": ["male", "female", "unknown"]})
    await client.aclose()


def test_placeholders_are_rejected():
    with pytest.raises(ValueError, match="real service host"):
        make_settings(vllm_base_url="https://your-vllm-host.example.com")
