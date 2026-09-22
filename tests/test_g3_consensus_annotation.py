from __future__ import annotations

import io
import json

import httpx
from PIL import Image

from g3_consensus_annotation.client import VllmClient
from g3_consensus_annotation.config import Settings
from g3_consensus_annotation.pipeline import run
from g3_consensus_annotation.taxonomy import TAXONOMY


def _image() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (200, 300), "steelblue").save(stream, "JPEG")
    return stream.getvalue()


def _attributes() -> dict:
    return {
        name: [] if item["type"] == "multi_label" else item["classes"][0]
        for name, item in TAXONOMY.items()
    }


def test_g3_runs_four_observers_consensus_then_text_caption():
    attrs = _attributes()
    responses = iter(
        [{"caption": f"Observer {index}.", "attributes": attrs} for index in range(4)]
        + [{"attributes": attrs}, {"caption": "A person wearing a jacket."}, {"caption_vi": "Một người mặc áo khoác."}]
    )
    schemas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, request=request, json={"access_token": "token", "expires_in": 3600})
        body = json.loads(request.content)
        schemas.append(body["response_format"]["json_schema"]["name"])
        return httpx.Response(200, request=request, json={"choices": [{"message": {"content": json.dumps(next(responses))}}]})

    settings = Settings("https://model.example.test/v1", "id", "secret", "project")
    with VllmClient(settings, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None) as client:
        result = run(_image(), client)
    assert result == {"caption": "A person wearing a jacket.", "caption_vi": "Một người mặc áo khoác.", "attributes": attrs}
    assert schemas == ["observer_1", "observer_2", "observer_3", "observer_4", "attribute_consensus", "caption", "caption_vietnamese"]
