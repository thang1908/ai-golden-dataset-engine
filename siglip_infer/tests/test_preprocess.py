"""Preprocessing is the half of the model contract that lives outside the graph.

⚠️ Every failure this file guards is silent: a wrong resize, a wrong
normalization, or an un-rotated phone photo all produce a finite unit-norm
embedding and a full set of confident attributes. Nothing raises. The results
are just quietly worse, and no downstream check can detect it.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from siglip_infer.preprocess import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PreprocessError,
    crop_to_box,
    load_image,
    preprocess_batch,
    preprocess_image,
)


def _image(width: int, height: int, color=(120, 30, 200)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def test_output_is_the_fixed_geometry_the_positional_table_was_folded_for():
    tensor = preprocess_image(_image(64, 300))
    assert tensor.shape == (3, INPUT_HEIGHT, INPUT_WIDTH) == (3, 384, 128)
    assert tensor.dtype == np.float16


def test_the_resize_squashes_and_does_not_preserve_aspect_ratio():
    # The model was trained on squashed 3:1 crops. A "helpful" letterbox or
    # aspect-preserving resize changes every embedding the bundle produces.
    # A 400x50 image is 8:1; under letterboxing it would occupy 16 of the 384
    # rows and the rest would be padding, so the corners test for exactly that.
    source = Image.new("RGB", (400, 50), (255, 0, 0))
    source.paste(Image.new("RGB", (200, 50), (0, 0, 255)), (200, 0))
    tensor = preprocess_image(source).astype(np.float32)

    assert tensor.shape == (3, 384, 128)
    top_left = tensor[:, 0, 0]
    bottom_right = tensor[:, 383, 127]
    assert top_left == pytest.approx([1.0, -1.0, -1.0], abs=1e-2)
    assert bottom_right == pytest.approx([-1.0, -1.0, 1.0], abs=1e-2)


def test_normalization_maps_black_to_minus_one_and_white_to_plus_one():
    black = preprocess_image(_image(10, 10, (0, 0, 0))).astype(np.float32)
    white = preprocess_image(_image(10, 10, (255, 255, 255))).astype(np.float32)
    assert np.allclose(black, -1.0, atol=1e-3)
    assert np.allclose(white, 1.0, atol=1e-3)


def test_channels_stay_in_rgb_order_through_the_transpose():
    # A CHW transpose that also flips channel order is BGR, which recolours
    # every crop and quietly ruins the colour attributes.
    tensor = preprocess_image(_image(10, 10, (255, 0, 0))).astype(np.float32)
    assert tensor[0].mean() == pytest.approx(1.0, abs=1e-2)
    assert tensor[1].mean() == pytest.approx(-1.0, abs=1e-2)
    assert tensor[2].mean() == pytest.approx(-1.0, abs=1e-2)


def test_grayscale_rgba_and_palette_images_all_load_as_rgb():
    for mode in ("L", "RGBA", "P", "CMYK", "I;16"):
        image = Image.new(mode, (20, 40))
        assert load_image(image).mode == "RGB", mode


def test_an_exif_rotated_photo_is_uprighted_before_resizing():
    # A sideways person is a different person to this model, and no stage after
    # preprocessing can recover the rotation.
    upright = Image.new("RGB", (40, 80))
    upright.putpixel((0, 0), (255, 0, 0))
    buffer = io.BytesIO()
    rotated = upright.transpose(Image.Transpose.ROTATE_90)
    exif = rotated.getexif()
    exif[0x0112] = 8  # orientation: rotate 270 CW to display
    rotated.save(buffer, format="JPEG", exif=exif)
    loaded = load_image(buffer.getvalue())
    assert (loaded.width, loaded.height) == (40, 80)


def test_a_path_a_pil_image_bytes_and_an_array_all_give_the_same_tensor(tmp_path):
    image = _image(60, 150, (10, 200, 90))
    path = tmp_path / "crop.png"
    image.save(path)

    from_path = preprocess_image(path)
    from_pil = preprocess_image(image)
    from_bytes = preprocess_image(path.read_bytes())
    from_array = preprocess_image(np.asarray(image))

    for other in (from_pil, from_bytes, from_array):
        assert np.array_equal(from_path, other)


def test_a_float_array_outside_zero_to_one_is_rejected_not_clipped():
    # A [0, 255] float array clipped to [0, 1] washes every crop pure white and
    # still returns a confident embedding.
    array = np.full((20, 20, 3), 200.0, dtype=np.float32)
    with pytest.raises(PreprocessError, match=r"\[0, 1\]"):
        preprocess_image(array)


def test_a_missing_file_names_the_path_rather_than_raising_deep_in_pillow(tmp_path):
    with pytest.raises(PreprocessError, match="not found"):
        preprocess_image(tmp_path / "absent.jpg")


def test_undecodable_bytes_fail_as_a_preprocess_error():
    with pytest.raises(PreprocessError):
        preprocess_image(b"not an image")


def test_a_normalized_box_crops_the_same_region_as_its_pixel_equivalent():
    image = _image(200, 400)
    fractional = crop_to_box(image, (0.25, 0.5, 0.75, 1.0))
    pixels = crop_to_box(image, (50, 200, 150, 400), normalized=False)
    assert fractional.size == pixels.size == (100, 200)


def test_a_fractional_box_expands_outward_so_it_never_cuts_into_the_person():
    image = _image(100, 100)
    cropped = crop_to_box(image, (10.4, 10.6, 50.2, 50.9), normalized=False)
    assert cropped.size == (41, 41)


def test_an_inverted_or_empty_box_is_an_error():
    image = _image(100, 100)
    with pytest.raises(PreprocessError):
        crop_to_box(image, (0.8, 0.1, 0.2, 0.9))
    with pytest.raises(PreprocessError):
        crop_to_box(image, (0.5, 0.5, 0.5, 0.9))


def test_a_normalized_box_outside_the_unit_square_is_an_error():
    with pytest.raises(PreprocessError, match=r"\[0, 1\]"):
        crop_to_box(_image(100, 100), (0.0, 0.0, 1.5, 1.0), normalized=True)


def test_a_one_pixel_image_still_preprocesses():
    assert preprocess_image(_image(1, 1)).shape == (3, 384, 128)


def test_batching_preserves_input_order():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    batch = preprocess_batch([_image(20, 40, color) for color in colors])
    assert batch.shape == (3, 3, 384, 128)
    for index in range(3):
        expected = preprocess_image(_image(20, 40, colors[index]))
        assert np.array_equal(batch[index], expected)


def test_a_box_list_that_does_not_match_the_image_list_is_an_error():
    with pytest.raises(PreprocessError, match="1 boxes for 2 images"):
        preprocess_batch([_image(10, 10), _image(10, 10)], boxes=[(0.0, 0.0, 1.0, 1.0)])


def test_an_empty_batch_is_an_error_rather_than_an_empty_tensor():
    with pytest.raises(PreprocessError):
        preprocess_batch([])
