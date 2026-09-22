"""SigLIP uses native decoding and retains all groups, including hidden display tiers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_openai_compatible import jpeg_bytes

from c2_siglip.__main__ import main
from c2_siglip.runner import SiglipLoadError, SiglipPredictor, validate_threshold, verify_bundle
from golden_dataset_harness.baselines.io import BaselineConfigError, run_images
from golden_dataset_harness.schemas.taxonomy import TAXONOMY


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_thresholds(value):
    with pytest.raises(BaselineConfigError):
        validate_threshold(value)


def test_bad_cli_threshold_fails_before_loading_model(tmp_path, capsys):
    image = tmp_path / "person.jpg"
    image.write_bytes(jpeg_bytes())
    assert main(["--image", str(image), "--threshold", "nan",
                 "--output-dir", str(tmp_path / "out")]) == 2
    assert "threshold" in capsys.readouterr().err


def test_vision_integrity_does_not_require_text_artifacts(tmp_path):
    models = tmp_path / "models"
    assets = tmp_path / "assets"
    models.mkdir()
    assets.mkdir()
    model = models / "siglip_vitb_attr.onnx"
    model.write_bytes(b"fake-artifact-for-integrity-test")
    schema = assets / "attribute_schema.json"
    schema.write_text(json.dumps({"attributes": [
        {"name": name, **definition} for name, definition in TAXONOMY.items()
    ]}))
    artifacts = {p.name: {"bytes": p.stat().st_size,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in (model, schema)}
    (models / "siglip_manifest.json").write_text(json.dumps({"artifacts": artifacts}))
    (assets / "attribute_taxonomy.json").write_text(json.dumps({
        "generated_from": {"sha256": artifacts[schema.name]["sha256"]},
    }))
    _, hashes = verify_bundle(tmp_path)
    assert hashes[model.name] == artifacts[model.name]["sha256"]
    model.write_bytes(b"tampered")
    with pytest.raises(SiglipLoadError, match="checksum"):
        verify_bundle(tmp_path)


@pytest.mark.asyncio
async def test_native_decoder_to_shared_jsonl(tmp_path):
    taxonomy_module = pytest.importorskip("siglip_infer.taxonomy")
    taxonomy = taxonomy_module.load_taxonomy()
    logits = [-20.0] * 200
    for attr in taxonomy:
        if not attr.is_multi_label:
            logits[attr.start] = 20.0
    for code in ("backpack", "handbag"):
        logits[taxonomy["bag_type"].index_of(code)] = 20.0
    predictions = taxonomy_module.decode_logits(logits, taxonomy)

    class Encoder:
        def encode(self, image):
            assert isinstance(image, bytes)
            return SimpleNamespace(attributes=predictions)

    predictor = SiglipPredictor(Encoder(), "siglip-test", {"test_only": True})
    image = tmp_path / "person.jpg"
    image.write_bytes(jpeg_bytes())
    summary = await run_images(predictor, [image], tmp_path / "out")
    row = json.loads((tmp_path / "out/predictions.jsonl").read_text())
    assert row["attributes"]["bag_type"] == ["backpack", "handbag"]
    assert row["attributes"]["carried_objects"] == []
    assert row["attributes"]["body_build"] == "heavy"  # hidden display tier retained
    assert len(row["attributes"]) == 21
    assert row["scores"]["carried_objects"] == {}
    assert row["scores"]["bag_type"]["backpack"] > 0.99
    assert summary["model_calls"] == 1
    assert summary["http_attempts"] == 0


@pytest.mark.siglip_model
@pytest.mark.asyncio
async def test_actual_onnx_smoke(tmp_path):
    pytest.importorskip("onnxruntime")
    vision = pytest.importorskip("siglip_infer.vision")
    if not Path(vision.DEFAULT_MODEL_PATH).is_file():
        pytest.skip("Local vision artifact is required")
    from c2_siglip.runner import run

    image = tmp_path / "synthetic.jpg"
    image.write_bytes(jpeg_bytes((128, 384)))
    summary = await run([image], tmp_path / "out", device="cpu")
    assert summary["success_count"] == 1
    row = json.loads((tmp_path / "out/predictions.jsonl").read_text())
    assert list(row["attributes"]) == list(TAXONOMY)
    assert summary["model_load_ms"] > 0
    assert summary["config"]["providers"] == ["CPUExecutionProvider"]
