"""Model provenance and artifact-integrity contracts."""

from __future__ import annotations

import hashlib
import json

import pytest
from siglip_infer.runtime import ModelUnavailable
from siglip_infer.vision import DEFAULT_MANIFEST_PATH, load_manifest, verify_manifest_artifacts


def test_the_bundle_pins_the_epoch_35_encoder_and_detached_head():
    manifest = load_manifest()

    assert manifest["model_version"] == "v2"
    assert manifest["checkpoint"]["epoch"] == 35
    assert manifest["attribute_head"]["source"] == "detached_checkpoint"
    assert manifest["attribute_head"]["architecture"] == [768, 512, 200]


def test_a_mixed_model_bundle_is_rejected_by_hash(tmp_path):
    models = tmp_path / "models"
    assets = tmp_path / "assets"
    models.mkdir()
    assets.mkdir()
    expected = b"epoch-35"
    (models / "siglip_text.onnx").write_bytes(b"epoch-34")
    manifest = {
        "artifacts": {
            "siglip_text.onnx": {
                "bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
            }
        }
    }
    manifest_path = models / "siglip_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelUnavailable, match="SHA-256"):
        verify_manifest_artifacts(manifest_path, bundle_root=tmp_path)


@pytest.mark.models
@pytest.mark.skipif(
    not all(
        (DEFAULT_MANIFEST_PATH.parent / name).is_file()
        for name in ("siglip_vitb_attr.onnx", "siglip_text.onnx", "siglip_tokenizer.json")
    ),
    reason="ONNX model files are not present in models/",
)
def test_every_packaged_model_artifact_matches_the_manifest():
    verify_manifest_artifacts()
