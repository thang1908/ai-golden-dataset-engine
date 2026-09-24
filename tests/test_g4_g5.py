from __future__ import annotations

import io
import json

import httpx
from PIL import Image

from g4_critic_verifier.client import VllmClient as G4Client
from g4_critic_verifier.config import Settings as G4Settings
from g4_critic_verifier.pipeline import run as run_g4
from g4_critic_verifier.taxonomy import TAXONOMY
from g5_qa_refinement.client import VllmClient as G5Client
from g5_qa_refinement.config import Settings as G5Settings
from g5_qa_refinement.pipeline import run as run_g5


def _image() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (100, 200), "steelblue").save(stream, "JPEG")
    return stream.getvalue()


def _attributes() -> dict:
    return {name: [] if item["type"] == "multi_label" else item["classes"][0] for name, item in TAXONOMY.items()}


def _client(client_type, settings_type, responses, names):
    received = []
    iterator = iter(responses)
    def handler(request):
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, request=request, json={"access_token": "token", "expires_in": 3600})
        body = json.loads(request.content)
        received.append(body["response_format"]["json_schema"]["name"])
        return httpx.Response(200, request=request, json={"choices": [{"message": {"content": json.dumps(next(iterator))}}]})
    settings = settings_type("https://model.example.test/v1", "id", "secret", "project")
    return client_type(settings, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None), received, names


def test_g4_accepts_first_verified_draft():
    attrs = _attributes()
    client, received, _ = _client(G4Client, G4Settings, [{"caption": "A person.", "attributes": attrs}, {"issues": []}, {"decision": "accept", "reasons": ["Supported."]}, {"caption_vi": "Một người."}], [])
    with client:
        assert run_g4(_image(), client) == {"caption": "A person.", "caption_vi": "Một người.", "attributes": attrs, "workflow_status": "accepted", "workflow_notes": []}
    assert received == ["draft_annotation", "critic_issues", "verification", "caption_vietnamese"]


def test_g5_accepts_first_question_answer_round():
    attrs = _attributes()
    client, received, _ = _client(G5Client, G5Settings, [{"caption": "A person.", "attributes": attrs}, {"questions": [{"id": "q1", "target": "caption", "question": "Is a person visible?"}]}, {"answers": [{"question_id": "q1", "answer": "yes", "evidence": "person visible", "determinable": True}]}, {"decision": "accept", "corrections": []}, {"caption_vi": "Một người."}], [])
    with client:
        assert run_g5(_image(), client) == {"caption": "A person.", "caption_vi": "Một người.", "attributes": attrs, "workflow_status": "accepted", "workflow_notes": []}
    assert received == ["draft_annotation", "verification_questions", "image_answers", "comparison", "caption_vietnamese"]
