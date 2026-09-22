"""Image preprocessing — the half of the contract that lives outside the graph.

The exported ONNX model starts at a normalized ``(batch, 3, 384, 128)`` tensor.
Everything before that is here, and it must match training exactly:

    RGB -> squash-resize to 384x128 (bicubic) -> /255 -> (x - 0.5) / 0.5

⚠️ Every plausible variation of this — letterboxing instead of squashing,
bilinear instead of bicubic, ImageNet mean/std instead of 0.5 — still produces a
finite unit-norm 768-d embedding and a full set of confident attributes. The
results are just quietly, unfalsifiably worse. There is no runtime error to
catch, which is why the constants live in one place and are asserted against the
model manifest at load time rather than passed around.

"Squash" means the aspect ratio is **not** preserved: a person crop is stretched
to 3:1. That is what the model was trained on, so a "helpful" aspect-preserving
resize is a bug, not an improvement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

#: Model input geometry, (height, width). Fixed at export: the positional
#: embedding table was folded to a 24x8 patch grid for exactly this size.
INPUT_HEIGHT = 384
INPUT_WIDTH = 128

PIXEL_MEAN = (0.5, 0.5, 0.5)
PIXEL_STD = (0.5, 0.5, 0.5)

#: The exported graph's input dtype. It casts to FP32 immediately inside the
#: graph, so the weights stay FP32 and this is a bandwidth choice, not a
#: precision policy.
INPUT_DTYPE = np.float16

#: What ``load_image`` accepts.
ImageSource: TypeAlias = "str | Path | Image.Image | NDArray[Any] | bytes"

_MEAN = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD = np.asarray(PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)


class PreprocessError(ValueError):
    """The input could not be turned into a valid model tensor."""


def load_image(source: ImageSource) -> Image.Image:
    """Coerce any supported source into an upright RGB Pillow image.

    Handles the cases a real deployment actually hits: PNG with alpha, 16-bit
    or palette images, grayscale, CMYK, and phone photos whose pixels are
    sideways until the EXIF orientation tag is applied.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, np.ndarray):
        image = _from_array(source)
    elif isinstance(source, bytes | bytearray | memoryview):
        import io

        try:
            image = Image.open(io.BytesIO(bytes(source)))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise PreprocessError(f"could not decode image bytes: {error}") from error
    elif isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise PreprocessError(f"image not found at {path}")
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB") if opened.mode != "RGB" else opened.copy()
        except (UnidentifiedImageError, OSError) as error:
            raise PreprocessError(f"could not read image at {path}: {error}") from error
    else:
        raise PreprocessError(f"unsupported image source type {type(source).__name__}")

    # EXIF orientation before anything else: a sideways person is a different
    # person to this model, and nothing downstream can recover the rotation.
    image = ImageOps.exif_transpose(image) or image
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise PreprocessError("image has a zero dimension")
    return image


def _from_array(array: NDArray[Any]) -> Image.Image:
    """Accept HWC uint8/float arrays, the shape an OpenCV or decoder hand-off has.

    Float arrays are assumed to be in ``[0, 1]`` — the common convention — and a
    float array outside that range is rejected rather than clipped, because
    silently clipping a ``[0, 255]`` float array would wash every crop white.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis].repeat(3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise PreprocessError(
            f"image array must be HxW, HxWx1, HxWx3 or HxWx4, got shape {array.shape}"
        )
    if array.shape[2] == 1:
        array = array.repeat(3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]

    if array.dtype == np.uint8:
        pixels = array
    elif np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and (finite.min() < -1e-6 or finite.max() > 1.0 + 1e-6):
            raise PreprocessError(
                "float image array must be in [0, 1]; "
                f"got range [{float(finite.min()):.3f}, {float(finite.max()):.3f}]"
            )
        pixels = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        raise PreprocessError(f"unsupported image array dtype {array.dtype}")
    return Image.fromarray(np.ascontiguousarray(pixels))


def crop_to_box(
    image: Image.Image,
    box: Sequence[float],
    *,
    normalized: bool | None = None,
) -> Image.Image:
    """Crop to a left/top/right/bottom box before resizing.

    ``normalized`` selects between fractional (0-1) and pixel coordinates. Left
    as ``None`` it is inferred — every coordinate in ``[0, 1]`` means
    fractional — which is right in practice and ambiguous for a box that is
    genuinely one pixel at the origin. Pass it explicitly when a caller knows.

    Bounds are expanded with floor/ceil rather than rounded, so a fractional
    detector box never crops *into* the person.
    """
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise PreprocessError("box must be four finite numbers (left, top, right, bottom)")
    left, top, right, bottom = (float(value) for value in box)
    if right <= left or bottom <= top:
        raise PreprocessError(f"box ({left}, {top}, {right}, {bottom}) is empty or inverted")

    if normalized is None:
        normalized = all(0.0 <= value <= 1.0 for value in (left, top, right, bottom))
    if normalized:
        if not all(0.0 <= value <= 1.0 for value in (left, top, right, bottom)):
            raise PreprocessError("normalized box coordinates must lie in [0, 1]")
        left, right = left * image.width, right * image.width
        top, bottom = top * image.height, bottom * image.height

    pixel_left = max(0, min(image.width - 1, math.floor(left)))
    pixel_top = max(0, min(image.height - 1, math.floor(top)))
    pixel_right = max(pixel_left + 1, min(image.width, math.ceil(right)))
    pixel_bottom = max(pixel_top + 1, min(image.height, math.ceil(bottom)))
    return image.crop((pixel_left, pixel_top, pixel_right, pixel_bottom))


def preprocess_image(
    source: ImageSource,
    *,
    box: Sequence[float] | None = None,
    normalized_box: bool | None = None,
) -> NDArray[np.float16]:
    """One image -> one ``(3, 384, 128)`` FP16 CHW tensor, ready to batch."""
    image = load_image(source)
    if box is not None:
        image = crop_to_box(image, box, normalized=normalized_box)

    # Pillow's resize takes (width, height); the model contract is (H, W).
    resized = image.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - _MEAN) / _STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=INPUT_DTYPE)


def preprocess_batch(
    sources: Sequence[ImageSource],
    *,
    boxes: Sequence[Sequence[float] | None] | None = None,
    normalized_box: bool | None = None,
) -> NDArray[np.float16]:
    """Many images -> one ``(N, 3, 384, 128)`` FP16 NCHW batch."""
    if not sources:
        raise PreprocessError("cannot preprocess an empty batch")
    if boxes is not None and len(boxes) != len(sources):
        raise PreprocessError(f"got {len(boxes)} boxes for {len(sources)} images")

    batch = np.empty((len(sources), 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=INPUT_DTYPE)
    for index, source in enumerate(sources):
        box = None if boxes is None else boxes[index]
        batch[index] = preprocess_image(source, box=box, normalized_box=normalized_box)
    return batch
