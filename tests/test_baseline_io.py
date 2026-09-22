"""Baseline outputs must retain failures and preserve the shared attribute contract."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_openai_compatible import jpeg_bytes

from golden_dataset_harness.baselines.contracts import InferenceResult
from golden_dataset_harness.baselines.io import BaselineConfigError, discover_images, run_images
from golden_dataset_harness.schemas.taxonomy import TAXONOMY


def complete_attributes():
    return {
        name: [] if definition["type"] == "multi_label" else definition["classes"][0]
        for name, definition in TAXONOMY.items()
    }


class FakePredictor:
    method = "c2_siglip"
    model_id = "test-only"
    metadata = {"test_only": True}
    http_attempts = 0

    def __init__(self, fail=False):
        self.model_calls = 0
        self.fail = fail

    async def predict(self, image):
        self.model_calls += 1
        if self.fail:
            raise RuntimeError("credential-MUST-NOT-APPEAR base64-private-image")
        return InferenceResult(attributes=complete_attributes())


def test_discovery_preserves_extensions_and_is_not_recursive(tmp_path):
    for name in ["b.png", "a.jpg", "a.png", "ignore.txt"]:
        (tmp_path / name).touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "hidden.jpg").touch()
    assert [p.name for p in discover_images(None, tmp_path)] == ["a.jpg", "a.png", "b.png"]


@pytest.mark.parametrize("image,directory", [(None, None), ("missing.jpg", None), (None, "absent")])
def test_invalid_sources(image, directory):
    with pytest.raises(BaselineConfigError):
        discover_images(Path(image) if image else None, Path(directory) if directory else None)


def test_missing_attribute_is_not_silently_filled():
    attributes = complete_attributes()
    del attributes["age"]
    with pytest.raises(ValidationError, match="age"):
        InferenceResult(attributes=attributes)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_are_not_serialized_as_predictions(score):
    with pytest.raises(ValidationError):
        InferenceResult(attributes=complete_attributes(), scores={"gender": {"male": score}})


@pytest.mark.asyncio
async def test_jsonl_records_all_inputs_and_counts_only_attempted_inference(tmp_path):
    image = tmp_path / "same.jpg"
    image.write_bytes(jpeg_bytes())
    broken = tmp_path / "same.png"
    broken.write_bytes(b"not-an-image")
    output = tmp_path / "output"
    summary = await run_images(FakePredictor(), [image, broken], output, model_load_ms=12)
    rows = [json.loads(row) for row in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == summary["num_inputs"] == 2
    assert summary["success_count"] == summary["error_count"] == 1
    assert summary["model_calls"] == 1
    assert summary["http_attempts"] == 0
    assert summary["model_load_ms"] == 12
    assert rows[0]["image_id"] != rows[1]["image_id"]
    assert rows[0]["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert rows[0]["attributes"] == complete_attributes()
    assert rows[0]["attributes"]["bag_type"] == []
    assert rows[1]["status"] == "error"
    assert rows[1]["attributes"] is None
    assert rows[1]["model_calls"] == 0
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.asyncio
async def test_failed_predictions_keep_rows_and_do_not_leak_errors(tmp_path):
    image = tmp_path / "person.jpg"
    image.write_bytes(jpeg_bytes())
    output = tmp_path / "output"
    summary = await run_images(FakePredictor(fail=True), [image], output)
    assert summary["success_count"] == 0
    assert summary["error_count"] == 1
    assert summary["latency_ms_success"] is None
    text = (output / "predictions.jsonl").read_text()
    assert "credential-MUST-NOT-APPEAR" not in text
    assert "base64-private-image" not in text
    assert json.loads(text)["model_calls"] == 1


@pytest.mark.asyncio
async def test_overwrite_is_explicit_and_does_not_append(tmp_path):
    image = tmp_path / "person.jpg"
    image.write_bytes(jpeg_bytes())
    output = tmp_path / "output"
    await run_images(FakePredictor(), [image], output)
    before = (output / "predictions.jsonl").read_bytes()
    with pytest.raises(BaselineConfigError, match="already exists"):
        await run_images(FakePredictor(), [image], output)
    assert (output / "predictions.jsonl").read_bytes() == before
    summary = await run_images(FakePredictor(), [image], output, overwrite=True)
    rows = (output / "predictions.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["run_id"] == summary["run_id"]


@pytest.mark.asyncio
async def test_interruption_keeps_completed_rows_and_removes_stale_summary(tmp_path):
    class InterruptedPredictor(FakePredictor):
        async def predict(self, image):
            if self.model_calls:
                raise asyncio.CancelledError()
            return await super().predict(image)

    images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for image in images:
        image.write_bytes(jpeg_bytes())
    output = tmp_path / "out"
    await run_images(FakePredictor(), images, output)
    with pytest.raises(asyncio.CancelledError):
        await run_images(InterruptedPredictor(), images, output, overwrite=True)
    assert not (output / "summary.json").exists()
    rows = (output / "predictions.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["image_id"] == "a.jpg"
