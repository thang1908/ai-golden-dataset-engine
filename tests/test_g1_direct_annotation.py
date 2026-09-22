from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
from PIL import Image

from g1_direct_annotation.cli import DEFAULT_OUTPUT_DIR, DEFAULT_TEST_DIR, REPOSITORY_ROOT, parser
from g1_direct_annotation.client import VllmClient, request_body
from g1_direct_annotation.config import DEFAULT_MODEL, Settings, load_settings
from g1_direct_annotation.dataset import generate_dataset, load_query_samples
from g1_direct_annotation.taxonomy import TAXONOMY


def _image() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (200, 300), "steelblue").save(stream, "JPEG")
    return stream.getvalue()


def _attributes() -> dict:
    return {
        name: [] if item["type"] == "multi_label" else item["classes"][0]
        for name, item in TAXONOMY.items()
    }


def _settings() -> Settings:
    return Settings(
        base_url="https://model.example.test/v1",
        client_id="id",
        client_secret="secret",
        project_id="project",
    )


def test_direct_request_is_medium_and_has_complete_schema():
    body = request_body(_settings(), "data:image/jpeg;base64,x")
    schema = body["response_format"]["json_schema"]["schema"]
    assert DEFAULT_MODEL == "v-llm-v1-medium"
    assert body["model"] == DEFAULT_MODEL
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert schema["required"] == ["caption", "attributes"]
    assert schema["properties"]["attributes"]["required"] == list(TAXONOMY)
    assert len(schema["properties"]["attributes"]["properties"]) == 21


def test_shared_vllm_dotenv_is_used_but_shared_model_is_ignored(tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "VLLM_BASE_URL=https://shared.example\n"
        "VLLM_CLIENT_ID=shared-id\n"
        "VLLM_CLIENT_SECRET=shared-secret\n"
        "VLLM_PROJECT_ID=shared-project\n"
        "VLLM_MODEL=v-llm-v1-large\n",
        encoding="utf-8",
    )
    settings = load_settings(dotenv_path=dotenv)
    assert settings.base_url == "https://shared.example"
    assert settings.client_id == "shared-id"
    assert settings.model == "v-llm-v1-medium"


def test_cli_defaults_use_external_sample_and_g1_output_folder():
    args = parser().parse_args([])
    assert args.test_dir == DEFAULT_TEST_DIR
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.test_dir == REPOSITORY_ROOT / "sample" / "test"


def test_direct_generation_uses_one_completion_and_validates_output():
    calls: list[str] = []
    expected = {"caption": "A person wearing a jacket.", "attributes": _attributes()}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(
                200, request=request, json={"access_token": "token", "expires_in": 3600}
            )
        body = json.loads(request.content)
        assert body["model"] == "v-llm-v1-medium"
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
        return httpx.Response(
            200, request=request, json={"choices": [{"message": {"content": json.dumps(expected)}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with VllmClient(_settings(), client=client, sleep=lambda _: None) as g1:
        assert g1.annotate(_image()) == expected
    assert calls == ["/oauth/token", "/v1/chat/completions"]


def test_dataset_rows_match_the_existing_prediction_shape(tmp_path: Path):
    test_dir = tmp_path / "test"
    image_path = test_dir / "images" / "p" / "query.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_image())
    headers = ["person_id", "person_key", *TAXONOMY]
    (test_dir / "attributes.tsv").write_text(
        "\t".join(headers) + "\n0\tp\t" + "\t".join("" for _ in TAXONOMY) + "\n", encoding="utf-8"
    )
    (test_dir / "pairs_en_medium.tsv").write_text(
        "person_id\tfilepath\tis_query\n0\timages/p/query.jpg\t1\n", encoding="utf-8"
    )
    samples = load_query_samples(test_dir)
    rows = generate_dataset(
        samples,
        lambda _: {"caption": "A person.", "caption_vi": "Một người.", "attributes": _attributes()},
        model="v-llm-v1-medium",
    )
    assert rows[0]["sample_id"] == "0"
    assert rows[0]["status"] == "success"
    assert rows[0]["caption"] == "A person."
    assert rows[0]["caption_vi"] == "Một người."
    assert list(rows[0]["attributes"]) == list(TAXONOMY)
    assert {
        "sample_id",
        "method",
        "model_id",
        "prompt_version",
        "caption",
        "caption_vi",
        "attributes",
        "latency_ms",
        "status",
    } <= set(rows[0])
