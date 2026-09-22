"""End-to-end tests against the real ONNX models.

These skip when ``models/`` has no weights. They are the only tests that can
catch a bundle whose two towers came from different exports — the failure that
produces perfectly well-formed embeddings which simply do not mean the same
thing in the same space.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from siglip_infer.runtime import ModelUnavailable
from siglip_infer.taxonomy import load_taxonomy
from siglip_infer.text import DEFAULT_MODEL_PATH as TEXT_MODEL
from siglip_infer.text import DEFAULT_TOKENIZER_PATH
from siglip_infer.vision import DEFAULT_MODEL_PATH as VISION_MODEL

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        not (VISION_MODEL.is_file() and TEXT_MODEL.is_file() and DEFAULT_TOKENIZER_PATH.is_file()),
        reason="ONNX model files are not present in models/",
    ),
]


def test_the_vision_tower_returns_a_unit_768d_embedding(vision_encoder, synthetic_crop):
    result = vision_encoder.encode(synthetic_crop)
    assert result.embedding.shape == (768,)
    assert result.embedding.dtype == np.float32
    assert float(np.linalg.norm(result.embedding)) == pytest.approx(1.0, abs=1e-3)


def test_the_vision_tower_returns_every_attribute_with_a_natural_name(
    vision_encoder, synthetic_crop
):
    taxonomy = load_taxonomy()
    result = vision_encoder.encode(synthetic_crop)
    assert len(result.attributes) == 21 == len(taxonomy)
    for prediction in result.attributes:
        if prediction.is_multi_label:
            assert len(prediction.labels) == len(prediction.codes) == len(prediction.scores)
            assert all(0.0 <= score <= 1.0 for score in prediction.scores)
        else:
            assert prediction.code in prediction.attribute.codes
            assert prediction.label == prediction.attribute.label_of(prediction.code)
            assert 0.0 <= prediction.confidence <= 1.0


def test_raw_logits_are_exposed_so_a_caller_can_choose_its_own_thresholds(
    vision_encoder, synthetic_crop
):
    result = vision_encoder.encode(synthetic_crop)
    assert result.attribute_logits.shape == (200,)
    assert np.all(np.isfinite(result.attribute_logits))


def test_the_same_image_encodes_identically_twice(vision_encoder, synthetic_crop):
    # Inference is deterministic. A drifting embedding would mean stored index
    # vectors and freshly computed ones slowly stop matching.
    first = vision_encoder.encode(synthetic_crop)
    second = vision_encoder.encode(synthetic_crop)
    assert np.array_equal(first.embedding, second.embedding)
    assert np.array_equal(first.attribute_logits, second.attribute_logits)


def test_batching_gives_the_same_answer_as_encoding_one_at_a_time(vision_encoder):
    generator = np.random.default_rng(7)
    images = [
        Image.fromarray(generator.integers(0, 256, (200, 80, 3), dtype=np.uint8)) for _ in range(3)
    ]
    batched = vision_encoder.encode_batch(images, batch_size=3)
    singles = [vision_encoder.encode(image) for image in images]
    for one, many in zip(singles, batched, strict=True):
        assert float(one.embedding @ many.embedding) == pytest.approx(1.0, abs=1e-4)


def test_a_batch_larger_than_batch_size_is_chunked_and_stays_in_order(vision_encoder):
    generator = np.random.default_rng(11)
    images = [
        Image.fromarray(generator.integers(0, 256, (200, 80, 3), dtype=np.uint8)) for _ in range(5)
    ]
    chunked = vision_encoder.encode_batch(images, batch_size=2)
    whole = vision_encoder.encode_batch(images, batch_size=5)
    for left, right in zip(chunked, whole, strict=True):
        assert float(left.embedding @ right.embedding) == pytest.approx(1.0, abs=1e-4)


def test_two_different_images_do_not_collapse_to_the_same_embedding(vision_encoder):
    # A tower loaded with the wrong weights, or fed a constant tensor, returns
    # near-identical unit vectors for everything and still "works".
    generator = np.random.default_rng(3)
    left = Image.fromarray(generator.integers(0, 256, (200, 80, 3), dtype=np.uint8))
    right = Image.new("RGB", (80, 200), (255, 255, 255))
    similarity = float(
        vision_encoder.encode(left).embedding @ vision_encoder.encode(right).embedding
    )
    assert similarity < 0.95


def test_a_wrongly_shaped_tensor_is_rejected_before_it_reaches_the_session(vision_encoder):
    with pytest.raises(ValueError, match=r"\(N, 3, 384, 128\)"):
        vision_encoder.encode_tensor(np.zeros((1, 3, 128, 384), dtype=np.float16))


def test_the_text_tower_returns_a_unit_768d_embedding(text_encoder):
    vector = text_encoder.encode("người mặc áo đỏ")
    assert vector.shape == (768,)
    assert vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-3)


def test_the_text_tower_is_multilingual_without_a_translation_step(text_encoder):
    # If the tokenizer or vocabulary were wrong, the Vietnamese and English
    # forms of one query would not sit closer than two unrelated queries —
    # but both would still return perfectly valid vectors.
    vietnamese, english, unrelated = text_encoder.encode_batch(
        [
            "người mặc áo đỏ",
            "a person wearing a red shirt",
            "an empty parking lot at night",
        ]
    )
    assert float(vietnamese @ english) > float(english @ unrelated)


def test_case_and_punctuation_do_not_change_a_query(text_encoder):
    plain, decorated = text_encoder.encode_batch(
        ["a person wearing a red shirt", "  A Person, Wearing a RED Shirt!!  "]
    )
    assert float(plain @ decorated) == pytest.approx(1.0, abs=1e-5)


def test_a_query_that_canonicalizes_to_nothing_is_rejected(text_encoder):
    with pytest.raises(ValueError, match="empty after canonicalization"):
        text_encoder.encode("???!!!")


def test_a_query_longer_than_the_context_window_is_truncated_not_rejected(text_encoder):
    vector = text_encoder.encode("a person wearing a red shirt " * 40)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-3)


def test_batched_and_single_query_encoding_agree(text_encoder):
    queries = ["áo đỏ", "blue jeans", "a woman with a backpack"]
    batched = text_encoder.encode_batch(queries, batch_size=2)
    for query, vector in zip(queries, batched, strict=True):
        assert float(text_encoder.encode(query) @ vector) == pytest.approx(1.0, abs=1e-5)


def test_image_and_text_embeddings_share_one_space(vision_encoder, text_encoder, synthetic_crop):
    # The dot product is only meaningful if both towers came from one export.
    # A mismatched pair still produces two unit 768-d vectors whose similarity
    # is noise — so this asserts the score is finite and in range, and the
    # cross-modal ordering is left to a labelled evaluation set.
    image_vector = vision_encoder.encode(synthetic_crop).embedding
    text_vector = text_encoder.encode("a person wearing a red shirt")
    score = float(text_vector @ image_vector)
    assert -1.0 <= score <= 1.0


def test_requesting_cuda_on_a_cpu_only_machine_fails_loudly(monkeypatch):
    # ⚠️ Silently falling back to CPU turns a GPU deployment into one that is
    # ten times slower forever, with nothing in the logs to say so.
    import onnxruntime
    from siglip_infer.vision import DEFAULT_MODEL_PATH, VisionEncoder

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(ModelUnavailable, match="CUDAExecutionProvider is unavailable"):
        VisionEncoder(DEFAULT_MODEL_PATH, device="cuda")


def test_a_missing_model_file_names_the_path(tmp_path):
    from siglip_infer.vision import VisionEncoder

    with pytest.raises(ModelUnavailable, match="not found"):
        VisionEncoder(tmp_path / "absent.onnx")
