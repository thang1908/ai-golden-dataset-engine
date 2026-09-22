"""Regression coverage for the 21-attribute SigLIP annotation contract."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_openai_compatible import jpeg_bytes, make_settings

from golden_dataset_harness.agents.confidence import confidence_scoring_node
from golden_dataset_harness.agents.grounding_agent import _generate_claims
from golden_dataset_harness.api.label_studio import (
    format_for_label_studio,
    parse_label_studio_export,
)
from golden_dataset_harness.api.main import _state, app
from golden_dataset_harness.models.openai_compatible import OpenAICompatibleVLM, VLMServiceError
from golden_dataset_harness.schemas.annotation import (
    AnnotationRecord,
    ConsensusResult,
    PersonAttributes,
    QualityJudgment,
)
from golden_dataset_harness.schemas.taxonomy import (
    TAXONOMY,
    attribute_cells,
    parse_attribute_cells,
)


def complete_attributes():
    return {
        name: [definition["classes"][0]] if definition["type"] == "multi_label"
        else definition["classes"][0]
        for name, definition in TAXONOMY.items()
    }


def record(**attrs):
    return AnnotationRecord(
        image_id="person-1", image_path="person-1.jpg", caption="A person.",
        attributes=PersonAttributes(**attrs), confidence=0.9,
        consensus_score=0.9, grounding_score=0.9, judge_score=0.9, model_version="test",
    )


def test_taxonomy_matches_siglip_source_and_order():
    assert len(TAXONOMY) == 21
    assert sum(len(d["classes"]) for d in TAXONOMY.values()) == 200
    assert [name for name, d in TAXONOMY.items() if d["type"] == "multi_label"] == [
        "bag_type", "bag_color", "other_accessories", "carried_objects",
    ]
    source = Path(__file__).resolve().parents[1] / "siglip_infer/assets/attribute_schema.json"
    if source.exists():
        schema = json.loads(source.read_text())
        assert list(TAXONOMY) == [a["name"] for a in schema["attributes"]]
        for attr in schema["attributes"]:
            assert TAXONOMY[attr["name"]]["type"] == attr["type"]
            assert TAXONOMY[attr["name"]]["classes"] == attr["classes"]


@pytest.mark.parametrize("attrs", [
    {"gender": "unknown"}, {"age": "teenager"}, {"upper_clothing_type": "t-shirt"},
    {"bag_type": "backpack"}, {"bag_type": ["none"]},
    {"bag_type": ["backpack", "backpack"]}, {"bag_type": [123]},
    {"other_accessories": ["robot"]}, {"hair": "long_hair"}, {"age": ["adult"]},
])
def test_invalid_attributes_are_rejected_at_model_boundary(attrs):
    with pytest.raises(ValidationError):
        PersonAttributes(**attrs)


def test_null_empty_and_unknown_are_distinct_and_round_trip():
    attrs = PersonAttributes(bag_type=[], bag_color=["unknown"], age="unknown")
    values = attrs.model_dump()
    assert values["carried_objects"] is None
    assert values["bag_type"] == []
    assert values["bag_color"] == ["unknown"]
    assert PersonAttributes.model_validate_json(attrs.model_dump_json()) == attrs
    cells = attribute_cells(values)
    assert cells["carried_objects"] == ""
    assert cells["bag_type"] == "none"
    assert cells["bag_color"] == "unknown"
    assert parse_attribute_cells(cells) == values


@pytest.mark.parametrize("cell", ["backpack|", "backpack|backpack", "none|backpack", "Backpack"])
def test_invalid_multilabel_table_cells_are_rejected(cell):
    with pytest.raises(ValueError):
        parse_attribute_cells({"bag_type": cell})


def test_grounding_covers_all_attributes_and_each_label():
    values = complete_attributes()
    values.update(age="adult", hair_length="long_hair", hair_texture="straight_hair",
                  hairstyle="ponytail", hair_color="black",
                  bag_type=["backpack", "handbag"], headwear="helmet",
                  eyewear="eyeglasses", face_mask="medical_mask")
    claims = _generate_claims("", PersonAttributes(**values))
    assert len(claims) == 22
    assert "The person has a backpack." in claims
    assert "The person has a handbag." in claims
    assert _generate_claims("", PersonAttributes()) == []
    empty_claims = _generate_claims("", PersonAttributes(bag_type=[], eyewear="none"))
    assert empty_claims == ["The person has no bag.", "The person wears no eyewear."]


@pytest.mark.asyncio
async def test_incomplete_attributes_cannot_be_auto_accepted():
    result = await confidence_scoring_node({
        "attributes": PersonAttributes(gender="female"),
        "consensus": ConsensusResult(caption="A person.", agreement_score=1),
        "grounding_score": 1, "judge_result": QualityJudgment(quality_score=1),
    }, weights={"judge": 0.4, "consensus": 0.3, "grounding": 0.3}, threshold=0.75)
    assert result["confidence"] == 1
    assert result["review_status"] == "pending_review"
    assert any("Unannotated attributes" in issue for issue in result["annotation"].issues)


def test_label_studio_round_trip_preserves_all_three_states():
    annotation = record(bag_type=["backpack", "handbag"], carried_objects=[],
                        age="unknown", bag_color=["black", "blue"])
    task = format_for_label_studio([annotation], "http://images.test")[0]
    task["annotations"] = [{"result": task["predictions"][0]["result"]}]
    review = parse_label_studio_export([task])[0]
    corrected = PersonAttributes(**review["corrected_attributes"])
    assert corrected == annotation.attributes


def test_api_contract_review_validation_and_siglip_export(monkeypatch):
    annotation = record(**complete_attributes())
    monkeypatch.setattr(_state, "records", {annotation.image_id: annotation})
    client = TestClient(app)
    assert client.get("/taxonomy").json()["total_classes"] == 200
    assert client.get("/export/siglip").json()["rows"] == []
    invalid = client.post("/review/person-1", json={
        "status": "human_approved", "corrected_attributes": {"gender": "robot"},
    })
    assert invalid.status_code == 422
    assert annotation.review_status == "pending_review"
    response = client.post("/review/person-1", json={
        "status": "human_approved", "corrected_caption": "Corrected caption.",
        "corrected_attributes": {"bag_type": ["backpack", "handbag"],
                                 "carried_objects": [], "body_build": None},
    })
    assert response.status_code == 200
    rows = client.get("/export/siglip").json()["rows"]
    assert len(rows) == 1
    assert rows[0]["bag_type"] == "backpack|handbag"
    assert rows[0]["carried_objects"] == "none"
    assert rows[0]["body_build"] == ""
    assert rows[0]["gender"] == "male"  # Unchanged by the partial correction.
    assert rows[0]["caption"] == "Corrected caption."
    properties = client.get("/openapi.json").json()["components"]["schemas"][
        "PersonAttributes"]["properties"]
    assert list(properties) == list(TAXONOMY)
    assert properties["bag_type"]["anyOf"][0]["type"] == "array"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [None, "scalar", "duplicate", "missing", "extra", "invalid"])
async def test_remote_extraction_full_contract(mutation):
    values = complete_attributes()
    values.update(bag_type=["backpack", "handbag"], bag_color=[], gender=None)
    if mutation == "scalar":
        values["bag_type"] = "backpack"
    elif mutation == "duplicate":
        values["bag_type"] = ["backpack", "backpack"]
    elif mutation == "missing":
        values.pop("age")
    elif mutation == "extra":
        values["bag"] = "backpack"
    elif mutation == "invalid":
        values["carried_objects"] = ["none"]

    class TokenProvider:
        async def get_token(self, force_refresh=False):
            return "test-token"

    def handler(request):
        payload = json.loads(request.content)
        schema = payload["response_format"]["json_schema"]["schema"]
        assert len(schema["required"]) == 21
        multi = schema["properties"]["bag_type"]["anyOf"][0]
        assert multi["type"] == "array"
        assert multi["uniqueItems"] is True
        assert multi["items"]["enum"] == TAXONOMY["bag_type"]["classes"]
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(values),
        }}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleVLM(make_settings(), TokenProvider(), client)
        if mutation:
            with pytest.raises(VLMServiceError):
                await model.extract_attributes(jpeg_bytes(), TAXONOMY)
        else:
            assert await model.extract_attributes(jpeg_bytes(), TAXONOMY) == values
