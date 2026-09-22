"""C1 production boundaries: config, OAuth, retry, strict response and atomic output."""

from __future__ import annotations

import base64
import io
import json
import os

import httpx
import pytest
from PIL import Image

from c1_vllm_medium.api import VllmAttributeClient, parse_annotation, request_body
from c1_vllm_medium.config import DEFAULT_MODEL, Settings, load_settings
from c1_vllm_medium.errors import ApiError, ConfigurationError, ResponseValidationError
from c1_vllm_medium.output import serialize_annotation, write_atomically
from c1_vllm_medium.taxonomy import TAXONOMY, json_schema, validate_attributes


def image_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (480, 640), (100, 120, 140)).save(stream, "JPEG")
    return stream.getvalue()


def complete_attributes() -> dict:
    return {
        name: [] if definition["type"] == "multi_label" else definition["classes"][0]
        for name, definition in TAXONOMY.items()
    }


def annotation(attributes: dict | None = None, caption: str = "A person wearing a jacket.") -> dict:
    return {"caption": caption, "attributes": attributes or complete_attributes()}


def settings(**overrides) -> Settings:
    value = {
        "base_url": "https://model.example.test/v1",
        "client_id": "client",
        "client_secret": "secret",
        "project_id": "project",
    }
    value.update(overrides)
    return Settings(**value)


def test_schema_contains_caption_and_exactly_21_attributes_and_multilabel_arrays():
    attributes = json_schema()
    body = request_body(settings(), "data:image/jpeg;base64,x")
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["caption", "attributes"]
    assert schema["properties"]["caption"]["minLength"] == 1
    assert attributes["required"] == list(TAXONOMY)
    assert len(attributes["properties"]) == 21
    assert attributes["additionalProperties"] is False
    assert attributes["properties"]["bag_type"]["anyOf"][0]["type"] == "array"
    assert attributes["properties"]["bag_type"]["anyOf"][0]["uniqueItems"] is True
    assert DEFAULT_MODEL == "v-llm-v1-medium"


@pytest.mark.parametrize("bad", [
    {"gender": "robot"},
    {**complete_attributes(), "extra": "no"},
    {key: value for key, value in complete_attributes().items() if key != "age"},
    {**complete_attributes(), "bag_type": "backpack"},
    {**complete_attributes(), "bag_type": ["backpack", "backpack"]},
])
def test_invalid_responses_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_attributes(bad)


def test_dotenv_environment_and_cli_precedence(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "C1_BASE_URL=https://file.example\nC1_CLIENT_ID=file-id\n"
        "C1_CLIENT_SECRET=file-secret\nC1_PROJECT_ID=file-project\nC1_MODEL=file-model\n"
    )
    monkeypatch.setenv("C1_MODEL", "environment-model")
    loaded = load_settings(
        dotenv_path=dotenv,
        overrides={"C1_BASE_URL": "https://cli.example", "C1_MODEL": "cli-model"},
    )
    assert loaded.base_url == "https://cli.example"
    assert loaded.model == "cli-model"
    assert loaded.client_secret == "file-secret"


def test_shared_vllm_configuration_uses_medium_model(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "VLLM_BASE_URL=https://shared.example\nVLLM_CLIENT_ID=shared-id\n"
        "VLLM_CLIENT_SECRET=shared-secret\nVLLM_PROJECT_ID=shared-project\n"
        "VLLM_MODEL=v-llm-v1-large\n"
    )
    loaded = load_settings(dotenv_path=dotenv)
    assert loaded.base_url == "https://shared.example"
    assert loaded.client_id == "shared-id"
    assert loaded.model == "v-llm-v1-medium"


def test_invalid_dotenv_and_config_do_not_expose_secret(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("invalid-line-with-secret\n")
    with pytest.raises(ConfigurationError) as error:
        load_settings(dotenv_path=dotenv)
    assert "secret" not in str(error.value).lower()


def test_documented_oauth_and_chat_request_with_json_schema():
    requests: list[httpx.Request] = []
    attributes = complete_attributes()
    attributes.update(gender=None, bag_type=["backpack", "handbag"], bag_color=["unknown"])
    expected = annotation(attributes, "A person carries a backpack and a handbag.")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/oauth/token"):
            assert json.loads(request.content) == {
                "client_id": "client", "client_secret": "secret", "project_id": "project",
            }
            return httpx.Response(200, request=request, json={
                "access_token": "access-token", "expires_in": 3600,
            })
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer access-token"
        assert payload["model"] == "v-llm-v1-medium"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["response_format"]["type"] == "json_schema"
        schema = payload["response_format"]["json_schema"]["schema"]
        assert payload["response_format"]["json_schema"]["name"] == "person_annotation"
        assert schema["required"] == ["caption", "attributes"]
        assert schema["properties"]["attributes"]["required"] == list(TAXONOMY)
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        assert image_url.startswith("data:image/jpeg;base64,")
        assert len(base64.b64decode(image_url.split(",", 1)[1])) <= 60_000
        return httpx.Response(200, request=request, json={
            "choices": [{"message": {"content": json.dumps(expected)}}],
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = VllmAttributeClient(
        settings(), client, sleep=lambda _: None
    ).annotate(image_bytes())
    assert result == expected
    assert [request.url.path for request in requests] == ["/oauth/token", "/v1/chat/completions"]


def test_401_refresh_and_transient_retry_are_bounded():
    token_calls = 0
    chat_calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls, chat_calls
        if request.url.path.endswith("/oauth/token"):
            token_calls += 1
            return httpx.Response(200, request=request, json={
                "access_token": f"token-{token_calls}", "expires_in": 3600,
            })
        chat_calls += 1
        if chat_calls == 1:
            return httpx.Response(401, request=request)
        if chat_calls == 2:
            return httpx.Response(429, request=request, headers={"retry-after": "0"})
        return httpx.Response(200, request=request, json={
            "choices": [{"message": {"content": json.dumps(annotation())}}],
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = VllmAttributeClient(
        settings(max_retries=3), client, sleep=delays.append
    ).annotate(image_bytes())
    assert result == annotation()
    assert token_calls == 2
    assert chat_calls == 3
    assert delays == [0.0]


def test_retry_exhaustion_and_response_errors_are_safe():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, request=request, json={
                "access_token": "token", "expires_in": 3600,
            })
        return httpx.Response(503, request=request, text="provider-secret-body")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError) as error:
        VllmAttributeClient(
            settings(max_retries=2), client, sleep=lambda _: None
        ).annotate(image_bytes())
    assert "provider-secret-body" not in str(error.value)

    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"caption":"", "attributes":{}}'}}]},
    )
    with pytest.raises(ResponseValidationError):
        parse_annotation(response)


def test_request_body_and_atomic_output(tmp_path):
    body = request_body(settings(), "data:image/jpeg;base64,x")
    assert body["model"] == "v-llm-v1-medium"
    assert body["response_format"]["json_schema"]["name"] == "person_annotation"
    target = tmp_path / "nested" / "annotation.json"
    expected = annotation()
    write_atomically(target, serialize_annotation(expected))
    assert json.loads(target.read_text()) == expected
    assert os.stat(target).st_mode & 0o777 == 0o600
