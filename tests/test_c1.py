"""Direct extraction: no extra AI stages, explicit model, real HTTP attempt accounting."""

import json

import httpx
import pytest
from test_baseline_io import complete_attributes
from test_openai_compatible import jpeg_bytes, make_settings

from c1_vllm_medium.__main__ import main
from c1_vllm_medium.runner import VLLMPredictor, load_settings
from golden_dataset_harness.baselines.io import BaselineConfigError, run_images
from golden_dataset_harness.models.oauth import OAuthTokenProvider
from golden_dataset_harness.models.openai_compatible import OpenAICompatibleVLM


def test_model_does_not_inherit_large(monkeypatch):
    monkeypatch.delenv("C1_VLLM_MODEL", raising=False)
    monkeypatch.setenv("VLLM_MODEL", "v-llm-v1-large")
    with pytest.raises(BaselineConfigError, match="Medium"):
        load_settings(None)
    with pytest.raises(BaselineConfigError, match="not v-llm-v1-large"):
        load_settings("v-llm-v1-large")


def test_medium_id_overrides_shared_model_and_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "https://model.example.test")
    monkeypatch.setenv("VLLM_CLIENT_ID", "client")
    monkeypatch.setenv("VLLM_CLIENT_SECRET", "secret")
    monkeypatch.setenv("VLLM_PROJECT_ID", "project")
    monkeypatch.setenv("VLLM_MODEL", "v-llm-v1-large")
    monkeypatch.setenv("C1_VLLM_MODEL", "medium-service-id")
    assert load_settings(None).vllm_model == "medium-service-id"
    assert load_settings("another-medium-id").vllm_model == "another-medium-id"


def test_setup_errors_do_not_echo_credentials(monkeypatch, tmp_path, capsys):
    image = tmp_path / "person.jpg"
    image.write_bytes(jpeg_bytes())
    monkeypatch.setenv("VLLM_BASE_URL", "private-host-and-secret-without-scheme")
    code = main(["--image", str(image), "--model", "medium-service-id",
                 "--output-dir", str(tmp_path / "out")])
    assert code == 2
    assert "private-host-and-secret" not in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_401", [False, True])
async def test_one_extraction_and_actual_http_attempts(tmp_path, monkeypatch, retry_401):
    requests = []
    chat_count = 0
    expected = complete_attributes()
    expected.update(gender=None, bag_type=["backpack", "handbag"], bag_color=["unknown"])

    def handler(request):
        nonlocal chat_count
        requests.append(request)
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, json={"access_token": "secret-token", "expires_in": 3600})
        chat_count += 1
        payload = json.loads(request.content)
        assert payload["model"] == "medium-service-id"
        assert payload["temperature"] == 0
        assert payload["response_format"]["json_schema"]["name"] == "person_attributes"
        assert len(payload["response_format"]["json_schema"]["schema"]["required"]) == 21
        if retry_401 and chat_count == 1:
            return httpx.Response(401)
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(expected),
        }}]})

    async def forbidden(*args, **kwargs):
        pytest.fail("C1 must not call caption, grounding or judge")

    for name in ("generate_caption", "verify_claim", "judge_quality"):
        monkeypatch.setattr(OpenAICompatibleVLM, name, forbidden)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        settings = make_settings(vllm_model="medium-service-id")
        model = OpenAICompatibleVLM(settings, OAuthTokenProvider(settings, client), client)
        predictor = VLLMPredictor(model)
        client.event_hooks["request"].append(predictor.count_request)
        image = tmp_path / "person.jpg"
        image.write_bytes(jpeg_bytes())
        output = tmp_path / "out"
        summary = await run_images(predictor, [image], output)
    assert summary["model_calls"] == 1
    assert summary["http_attempts"] == (2 if retry_401 else 1)
    row = json.loads((output / "predictions.jsonl").read_text())
    assert row["attributes"] == expected
    assert row["scores"] is None
    assert "confidence" not in row
    assert "secret-token" not in (output / "summary.json").read_text()
    assert summary["config"]["prompt_sha256"]


@pytest.mark.asyncio
async def test_invalid_remote_output_is_a_recorded_failure(tmp_path):
    class TokenProvider:
        async def get_token(self, **kwargs):
            return "test-token"

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "content": '{"gender":"female"}',
        }}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleVLM(
            make_settings(vllm_model="medium-test"), TokenProvider(), client,
        )
        predictor = VLLMPredictor(model)
        client.event_hooks["request"].append(predictor.count_request)
        image = tmp_path / "person.jpg"
        image.write_bytes(jpeg_bytes())
        summary = await run_images(predictor, [image], tmp_path / "out")
    assert summary["error_count"] == 1
    assert summary["model_calls"] == summary["http_attempts"] == 1
