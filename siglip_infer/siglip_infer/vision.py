"""The vision tower: one person crop in, one embedding and 21 attributes out.

    image -> preprocess (384x128 squash, bicubic, mean/std 0.5)
          -> ONNX (SigLIP2 ViT-B/16 + MLP-512 attribute head)
          -> L2-normalized 768-d embedding + 200 attribute logits
          -> decoded attributes with natural class names

Normalization happens *inside* the exported graph, and so does the attribute
head. This module adds no arithmetic to the model's outputs — it validates them
and names them. If the embedding comes back non-unit, that is a model-contract
mismatch, and repairing it here would hide the mismatch rather than fix it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from siglip_infer.preprocess import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    ImageSource,
    preprocess_batch,
)
from siglip_infer.runtime import Device, ModelUnavailable, build_session, describe_io
from siglip_infer.taxonomy import (
    PredictedAttributes,
    Taxonomy,
    decode_logits,
    load_taxonomy,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BUNDLE_ROOT / "models" / "siglip_vitb_attr.onnx"
DEFAULT_MANIFEST_PATH = BUNDLE_ROOT / "models" / "siglip_manifest.json"

EMBEDDING_DIM = 768
ATTRIBUTE_LOGITS = 200

#: How far the returned embedding may sit from unit length before the model is
#: declared broken. Generous enough for FP16 accumulation, tight enough that a
#: missing normalization node (which lands around 10-20) cannot pass.
UNIT_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class VisionResult:
    """One crop's embedding and attributes."""

    embedding: NDArray[np.float32]
    """L2-normalized 768-d vector. Cosine similarity is a plain dot product."""

    attributes: PredictedAttributes
    """All 21 attributes, decoded with natural class names."""

    attribute_logits: NDArray[np.float32]
    """The raw 200 logits, for callers that want their own thresholds."""

    def to_dict(self, *, include_embedding: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"attributes": self.attributes.to_dict()}
        if include_embedding:
            payload["embedding"] = [float(value) for value in self.embedding]
            payload["embedding_dim"] = int(self.embedding.shape[0])
        return payload


class VisionEncoder:
    """The SigLIP vision tower plus attribute head, on ONNX Runtime.

    Loaded once and reused; the session is safe to call from several threads.

        encoder = VisionEncoder()
        result = encoder.encode("crop.jpg")
        result.embedding                       # (768,) float32, unit norm
        result.attributes["gender"].label      # "Female"
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        *,
        device: Device | str = Device.AUTO,
        taxonomy: Taxonomy | None = None,
        intra_op_threads: int | None = None,
        multilabel_threshold: float | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._taxonomy = taxonomy or load_taxonomy()
        self._multilabel_threshold = multilabel_threshold
        self._session = build_session(
            self._model_path, device=device, intra_op_threads=intra_op_threads
        )
        self._validate_contract()

    @property
    def taxonomy(self) -> Taxonomy:
        return self._taxonomy

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def model_path(self) -> Path:
        return self._model_path

    def _validate_contract(self) -> None:
        """Fail at load time on a model that does not match this code.

        ⚠️ Every check here guards a failure that otherwise produces plausible
        output: a differently shaped embedding still ranks, and a differently
        sized logit vector still decodes into confident-looking attributes.
        """
        io = describe_io(self._session)
        inputs, outputs = io["inputs"], io["outputs"]

        if len(inputs) != 1 or inputs[0]["name"] != "image":
            raise ModelUnavailable(f"vision model must take one input named 'image'; got {inputs}")
        shape = inputs[0]["shape"]
        if len(shape) != 4 or list(shape[1:]) != [3, INPUT_HEIGHT, INPUT_WIDTH]:
            raise ModelUnavailable(
                f"vision input must be (batch, 3, {INPUT_HEIGHT}, {INPUT_WIDTH}); got {shape}"
            )
        if inputs[0]["type"] != "tensor(float16)":
            raise ModelUnavailable(f"vision input must be float16; got {inputs[0]['type']}")

        names = [item["name"] for item in outputs]
        if names != ["embedding", "attr_logits"]:
            raise ModelUnavailable(
                f"vision model must output ['embedding', 'attr_logits']; got {names}"
            )
        embedding_shape = outputs[0]["shape"]
        logits_shape = outputs[1]["shape"]
        if len(embedding_shape) != 2 or embedding_shape[1] != EMBEDDING_DIM:
            raise ModelUnavailable(
                f"embedding output must be (batch, {EMBEDDING_DIM}); got {embedding_shape}"
            )
        if len(logits_shape) != 2 or logits_shape[1] != ATTRIBUTE_LOGITS:
            raise ModelUnavailable(
                f"attr_logits output must be (batch, {ATTRIBUTE_LOGITS}); got {logits_shape}"
            )
        if logits_shape[1] != self._taxonomy.total_logits:
            raise ModelUnavailable(
                f"model emits {logits_shape[1]} logits but the taxonomy describes "
                f"{self._taxonomy.total_logits}"
            )

    def encode(
        self,
        source: ImageSource,
        *,
        box: Sequence[float] | None = None,
        normalized_box: bool | None = None,
    ) -> VisionResult:
        """Encode one image, optionally cropping to ``box`` first."""
        return self.encode_batch(
            [source],
            boxes=None if box is None else [box],
            normalized_box=normalized_box,
        )[0]

    def encode_batch(
        self,
        sources: Sequence[ImageSource],
        *,
        boxes: Sequence[Sequence[float] | None] | None = None,
        normalized_box: bool | None = None,
        batch_size: int = 16,
    ) -> list[VisionResult]:
        """Encode many images, in chunks of ``batch_size``."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        tensors = preprocess_batch(sources, boxes=boxes, normalized_box=normalized_box)
        results: list[VisionResult] = []
        for start in range(0, len(tensors), batch_size):
            chunk = np.ascontiguousarray(tensors[start : start + batch_size])
            results.extend(self.encode_tensor(chunk))
        return results

    def encode_tensor(self, batch: NDArray[Any]) -> list[VisionResult]:
        """Run an already-preprocessed ``(N, 3, 384, 128)`` batch.

        The escape hatch for callers with their own decode pipeline. The tensor
        must already carry this model's preprocessing — see ``preprocess.py``.
        """
        array = np.asarray(batch)
        if array.ndim == 3:
            array = array[np.newaxis, ...]
        if array.ndim != 4 or array.shape[1:] != (3, INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(
                f"batch must be (N, 3, {INPUT_HEIGHT}, {INPUT_WIDTH}); got {array.shape}"
            )
        array = np.ascontiguousarray(array, dtype=np.float16)

        embeddings, logits = self._session.run(["embedding", "attr_logits"], {"image": array})
        embeddings = np.asarray(embeddings, dtype=np.float32)
        logits = np.asarray(logits, dtype=np.float32)
        if embeddings.shape != (len(array), EMBEDDING_DIM):
            raise ModelUnavailable(
                f"model returned embeddings of shape {embeddings.shape}, "
                f"expected {(len(array), EMBEDDING_DIM)}"
            )
        if logits.shape != (len(array), ATTRIBUTE_LOGITS):
            raise ModelUnavailable(
                f"model returned logits of shape {logits.shape}, "
                f"expected {(len(array), ATTRIBUTE_LOGITS)}"
            )

        results: list[VisionResult] = []
        for embedding, logit_row in zip(embeddings, logits, strict=True):
            _check_embedding(embedding)
            results.append(
                VisionResult(
                    embedding=np.ascontiguousarray(embedding),
                    attributes=decode_logits(
                        logit_row.tolist(),
                        self._taxonomy,
                        multilabel_threshold=self._multilabel_threshold,
                    ),
                    attribute_logits=np.ascontiguousarray(logit_row),
                )
            )
        return results

    def close(self) -> None:
        self._session = None

    def __enter__(self) -> VisionEncoder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _check_embedding(embedding: NDArray[np.float32]) -> None:
    if not np.all(np.isfinite(embedding)):
        raise ModelUnavailable("vision tower produced non-finite embedding values")
    norm = float(np.linalg.norm(embedding))
    if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        # ⚠️ Normalization is inside the exported graph. Renormalizing here
        # would hide a model-contract mismatch behind a plausible vector.
        raise ModelUnavailable(f"vision tower produced a non-unit embedding (norm {norm:.6f})")


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """The export manifest: checkpoint provenance and the preprocessing contract."""
    source = Path(path)
    if not source.is_file():
        raise ModelUnavailable(f"model manifest not found at {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def verify_manifest_artifacts(
    path: Path | str = DEFAULT_MANIFEST_PATH,
    *,
    bundle_root: Path | str = BUNDLE_ROOT,
) -> dict[str, Any]:
    """Verify every packaged artifact against the export manifest."""
    manifest = load_manifest(path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ModelUnavailable("model manifest has no artifact checksums")

    root = Path(bundle_root)
    for name, expected in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ModelUnavailable(f"invalid artifact name in model manifest: {name!r}")
        if not isinstance(expected, dict):
            raise ModelUnavailable(f"invalid manifest entry for artifact {name!r}")

        directory = "assets" if name == "attribute_schema.json" else "models"
        artifact = root / directory / name
        if not artifact.is_file():
            raise ModelUnavailable(f"manifest artifact not found at {artifact}")

        expected_bytes = expected.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ModelUnavailable(f"invalid byte count for manifest artifact {name!r}")
        actual_bytes = artifact.stat().st_size
        if actual_bytes != expected_bytes:
            raise ModelUnavailable(
                f"manifest artifact {name!r} has {actual_bytes} bytes, expected {expected_bytes}"
            )

        expected_sha = expected.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ModelUnavailable(f"invalid SHA-256 for manifest artifact {name!r}")
        digest = hashlib.sha256()
        with artifact.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            # ⚠️ A mismatched text/vision pair still emits valid unit vectors,
            # but cosine scores no longer share a meaningful embedding space.
            raise ModelUnavailable(
                f"manifest artifact {name!r} has SHA-256 {actual_sha}, expected {expected_sha}"
            )
    return manifest
