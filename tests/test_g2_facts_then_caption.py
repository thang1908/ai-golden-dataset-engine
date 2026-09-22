from __future__ import annotations

import io
import json

import httpx
from PIL import Image

from g2_facts_then_caption.client import VllmClient
from g2_facts_then_caption.config import Settings
from g2_facts_then_caption.pipeline import run
from g2_facts_then_caption.taxonomy import TAXONOMY


def _image() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (200, 300), "steelblue").save(stream, "JPEG")
    return stream.getvalue()


def _settings() -> Settings:
    return Settings(
        base_url="https://model.example.test/v1",
        client_id="id",
        client_secret="secret",
        project_id="project",
    )


def _attributes() -> dict:
    return {
        name: [] if item["type"] == "multi_label" else item["classes"][0]
        for name, item in TAXONOMY.items()
    }


def test_g2_runs_visual_analysis_facts_then_text_only_caption():
    stages: list[dict] = []
    responses = iter(
        [
            {"observations": ["A visible jacket."], "uncertainties": []},
            {"facts": ["The person wears a jacket."], "attributes": _attributes()},
            {"caption": "A person wearing a jacket."},
            {"caption_vi": "Một người mặc áo khoác."},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, request=request, json={"access_token": "token", "expires_in": 3600})
        body = json.loads(request.content)
        stages.append(body)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": json.dumps(next(responses))}}]},
        )

    transport = httpx.MockTransport(handler)
    with VllmClient(_settings(), client=httpx.Client(transport=transport), sleep=lambda _: None) as client:
        result = run(_image(), client)

    assert result == {"caption": "A person wearing a jacket.", "caption_vi": "Một người mặc áo khoác.", "attributes": _attributes()}
    assert [item["response_format"]["json_schema"]["name"] for item in stages] == [
        "visual_analysis",
        "structured_facts",
        "caption",
        "caption_vietnamese",
    ]
    assert len(stages[0]["messages"][0]["content"]) == 2
    assert len(stages[1]["messages"][0]["content"]) == 2
    assert len(stages[2]["messages"][0]["content"]) == 1
