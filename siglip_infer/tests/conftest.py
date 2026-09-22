"""Test fixtures.

Tests split into two groups. Everything that can run without the 1.5 GB of
model weights does — taxonomy parsing, preprocessing geometry, canonicalization
— so a teammate can check out the bundle, run pytest, and get real coverage
before the weights finish copying. Tests that need the models are marked
``models`` and skip cleanly when the files are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT))

from siglip_infer.text import DEFAULT_MODEL_PATH as TEXT_MODEL  # noqa: E402
from siglip_infer.text import DEFAULT_TOKENIZER_PATH  # noqa: E402
from siglip_infer.vision import DEFAULT_MODEL_PATH as VISION_MODEL  # noqa: E402

_MODELS_PRESENT = (
    VISION_MODEL.is_file() and TEXT_MODEL.is_file() and DEFAULT_TOKENIZER_PATH.is_file()
)


@pytest.fixture(scope="session")
def vision_encoder():
    if not _MODELS_PRESENT:
        pytest.skip("ONNX model files are not present in models/")
    from siglip_infer.vision import VisionEncoder

    return VisionEncoder(device="cpu", intra_op_threads=4)


@pytest.fixture(scope="session")
def text_encoder():
    if not _MODELS_PRESENT:
        pytest.skip("ONNX model files are not present in models/")
    from siglip_infer.text import TextEncoder

    return TextEncoder(device="cpu", intra_op_threads=4)


@pytest.fixture(scope="session")
def synthetic_crop() -> Image.Image:
    """A deterministic stand-in for a person crop.

    Not a person, so nothing may assert on *which* attributes come back — only
    that the shape of the answer is right and that it is stable.
    """
    generator = np.random.default_rng(0x51_61_4C_49_50)
    return Image.fromarray(generator.integers(0, 256, size=(256, 96, 3), dtype=np.uint8))
