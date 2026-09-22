"""Use the existing bundle without altering its model, preprocessing or decoder."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from golden_dataset_harness.baselines.contracts import InferenceResult
from golden_dataset_harness.baselines.io import (
    BaselineConfigError,
    check_output,
    package_versions,
    run_images,
    sha256_file,
)
from golden_dataset_harness.schemas.taxonomy import TAXONOMY


class SiglipLoadError(RuntimeError):
    """Safe explanation of a failed bundle/runtime check."""


def validate_threshold(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise BaselineConfigError("--threshold must be a finite number between 0 and 1")
    return value


def verify_bundle(root: Path) -> tuple[dict, dict[str, str]]:
    """Verify only the vision artifact and schema; do not require the text tower."""
    manifest_path = root / "models" / "siglip_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = {"manifest_sha256": sha256_file(manifest_path)}
        for relative in ("models/siglip_vitb_attr.onnx", "assets/attribute_schema.json"):
            path = root / relative
            expected = manifest["artifacts"][path.name]
            digest = sha256_file(path)
            if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
                raise SiglipLoadError(f"Artifact checksum mismatch: {path.name}")
            hashes[path.name] = digest
        schema = json.loads((root / "assets/attribute_schema.json").read_text())
        taxonomy_path = root / "assets/attribute_taxonomy.json"
        taxonomy = json.loads(taxonomy_path.read_text())
        hashes["taxonomy_sha256"] = sha256_file(taxonomy_path)
        if taxonomy["generated_from"]["sha256"] != hashes["attribute_schema.json"]:
            raise SiglipLoadError("SigLIP taxonomy provenance does not match the schema")
        if [a["name"] for a in schema["attributes"]] != list(TAXONOMY):
            raise SiglipLoadError("SigLIP schema attribute order differs from the harness")
        for entry in schema["attributes"]:
            definition = TAXONOMY[entry["name"]]
            if entry["type"] != definition["type"] or entry["classes"] != definition["classes"]:
                raise SiglipLoadError("SigLIP class codes differ from the harness")
        return manifest, hashes
    except SiglipLoadError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SiglipLoadError("Missing or invalid SigLIP vision artifacts/manifest") from exc


class SiglipPredictor:
    method = "c2_siglip"
    http_attempts = 0

    def __init__(self, encoder, model_id: str, metadata: dict):
        self.encoder = encoder
        self.model_id = model_id
        self.metadata = metadata
        self.model_calls = 0

    async def predict(self, image: bytes) -> InferenceResult:
        self.model_calls += 1
        result = self.encoder.encode(image)
        attributes = {}
        scores = {}
        for prediction in result.attributes:
            if prediction.is_multi_label:
                attributes[prediction.name] = list(prediction.codes)
                scores[prediction.name] = dict(
                    zip(prediction.codes, prediction.scores, strict=True)
                )
            else:
                attributes[prediction.name] = prediction.code
                scores[prediction.name] = {prediction.code: prediction.confidence}
        return InferenceResult(attributes=attributes, scores=scores)

    def close(self) -> None:
        self.encoder.close()


def load_predictor(device: str, threshold: float) -> SiglipPredictor:
    validate_threshold(threshold)
    try:
        from siglip_infer.vision import BUNDLE_ROOT, VisionEncoder
    except ImportError as exc:
        raise SiglipLoadError(
            "Install the local bundle first: python -m pip install -e ./siglip_infer"
        ) from exc
    manifest, hashes = verify_bundle(BUNDLE_ROOT)
    try:
        encoder = VisionEncoder(device=device, multilabel_threshold=threshold)
        if list(encoder.taxonomy.names) != list(TAXONOMY):
            raise SiglipLoadError("Decoded taxonomy attribute order differs from the harness")
        for name, definition in TAXONOMY.items():
            attr = encoder.taxonomy[name]
            if list(attr.codes) != definition["classes"] or attr.kind.value != definition["type"]:
                raise SiglipLoadError("Decoded taxonomy class order differs from the harness")
    except Exception as exc:
        if "encoder" in locals():
            encoder.close()
        if isinstance(exc, SiglipLoadError):
            raise
        raise SiglipLoadError(
            "Unable to load SigLIP; check the ONNX runtime, model and requested device"
        ) from exc
    return SiglipPredictor(encoder, f"siglip-{manifest['model_version']}", {
        "device_requested": device, "providers": encoder.providers,
        "multilabel_threshold": threshold, "model_artifacts": hashes,
        "preprocessing": manifest["vision"]["preprocess"],
        "scores": "uncalibrated softmax winner / selected sigmoid classes",
        "packages": package_versions(["siglip-infer", "onnxruntime", "numpy", "Pillow"]),
    })


async def run(
    paths: list[Path], output_dir: Path, *, device: str = "cpu", threshold: float = 0.2,
    overwrite: bool = False,
) -> dict:
    check_output(output_dir, overwrite)
    validate_threshold(threshold)
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    predictor = load_predictor(device, threshold)
    try:
        return await run_images(
            predictor, paths, output_dir, overwrite=overwrite,
            model_load_ms=(time.perf_counter() - start) * 1000, started_at=started_at,
        )
    finally:
        predictor.close()
