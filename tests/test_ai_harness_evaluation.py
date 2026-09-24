from __future__ import annotations

import json
from pathlib import Path

from AI_harness_evaluation.config import Settings
from AI_harness_evaluation.contracts import (
    ATTRIBUTE_FIELDS,
    caption_factuality_schema,
    validate_caption_attribute_evaluation,
)
from AI_harness_evaluation.dataset import load_tasks
from AI_harness_evaluation.gemini_client import GeminiJudge
from AI_harness_evaluation.pipeline import evaluate_task


def _write_fixture(root: Path) -> tuple[Path, Path]:
    test_dir = root / "sample" / "test"
    image = test_dir / "images" / "person-a" / "query.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-decoded-by-the-fake-client")
    review = root / "output" / "review"
    review.mkdir(parents=True)
    (review / "captions_merged.csv").write_text(
        "sample_id,image_path\n0,images/person-a/query.jpg\n", encoding="utf-8"
    )
    attributes = ["person_id", *ATTRIBUTE_FIELDS]
    (test_dir / "attributes.tsv").write_text(
        "\t".join(attributes) + "\n" + "\t".join(["0", *("none" for _ in ATTRIBUTE_FIELDS)]) + "\n",
        encoding="utf-8",
    )
    output = root / "output" / "g1"
    output.mkdir(parents=True)
    prediction = {
        "sample_id": "0", "status": "success", "caption": "A person wearing a blue shirt.",
        "caption_vi": "Một người mặc áo xanh.",
        "attributes": {field: "none" for field in ATTRIBUTE_FIELDS},
    }
    (output / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    return test_dir, root / "output"


def _caption_payload() -> dict:
    return {
        "is_correct": True,
        "needs_human_review": False,
        "note": "Các chi tiết được nêu đều phù hợp với ảnh.",
    }


def _caption_attribute_payload() -> dict:
    return {
        field: {
            "mentioned": field == "upper_clothing_color",
            "is_correct": True if field == "upper_clothing_color" else None,
            "note": "Màu áo phù hợp nhãn." if field == "upper_clothing_color" else "Caption không nêu thuộc tính này.",
        }
        for field in ATTRIBUTE_FIELDS
    }


def test_task_join_and_v3_two_judge_calls_with_mocked_transport(tmp_path: Path) -> None:
    test_dir, outputs_root = _write_fixture(tmp_path)
    task = load_tasks(test_dir, outputs_root, ("g1",))[0]
    calls: list[tuple[str, bytes | None, str | None, dict]] = []

    def generate(prompt: str, image: bytes | None, mime_type: str | None, schema: dict) -> str:
        calls.append((prompt, image, mime_type, schema))
        if image is not None:
            assert image == b"not-decoded-by-the-fake-client"
            assert mime_type == "image/jpeg"
            assert "Golden attributes" not in prompt
            assert set(schema["properties"]) == {"is_correct", "needs_human_review", "note"}
            return json.dumps(_caption_payload())
        assert mime_type is None
        assert "Golden attributes" in prompt
        assert "NO image" in prompt
        assert set(schema["properties"]) == set(ATTRIBUTE_FIELDS)
        variants = schema["properties"]["age"]["anyOf"]
        assert {variant["properties"]["mentioned"]["enum"][0] for variant in variants} == {True, False}
        assert {variant["properties"]["is_correct"]["type"] for variant in variants} == {"boolean", "null"}
        return json.dumps(_caption_attribute_payload())

    row = evaluate_task(
        task, GeminiJudge(Settings(api_key="test-key"), generate=generate), model="gemini-2.5-pro"
    )
    assert row["status"] == "success"
    assert row["evaluation_schema_version"] == "3"
    assert len(calls) == 2
    assert row["caption_evaluation"]["is_correct"] is True
    assert row["caption_attribute_evaluation"]["age"] == {
        "mentioned": False, "is_correct": None, "note": "Caption không nêu thuộc tính này.",
    }
    assert row["caption_attribute_summary"] == {
        "mentioned": 1, "not_mentioned": 20, "correct": 1, "incorrect": 0, "evaluated": 1,
    }
    assert row["prediction"]["attributes"] == {field: "none" for field in ATTRIBUTE_FIELDS}


def test_unmentioned_caption_attribute_must_be_null() -> None:
    invalid = _caption_attribute_payload()
    invalid["age"]["is_correct"] = False
    try:
        validate_caption_attribute_evaluation(invalid)
    except Exception as exc:
        assert "not mentioned" in str(exc)
    else:
        raise AssertionError("Expected invalid caption attribute to be rejected")


def test_mentioned_caption_attribute_must_be_boolean() -> None:
    invalid = _caption_attribute_payload()
    invalid["upper_clothing_color"]["is_correct"] = None
    try:
        validate_caption_attribute_evaluation(invalid)
    except Exception as exc:
        assert "true or false when it is mentioned" in str(exc)
    else:
        raise AssertionError("Expected invalid caption attribute to be rejected")


def test_judge_retries_transient_server_error() -> None:
    class ServerError(Exception):
        code = 500

    calls = 0

    def generate(*_args: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ServerError()
        return json.dumps(_caption_payload())

    judge = GeminiJudge(
        Settings(api_key="test-key", max_retries=1), generate=generate, sleep=lambda _seconds: None,
    )
    result = judge.complete(
        prompt="test", image=b"test", image_path_name="test.jpg",
        response_schema=caption_factuality_schema(),
    )
    assert calls == 2
    assert result["is_correct"] is True
