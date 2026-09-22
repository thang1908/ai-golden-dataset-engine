"""Standalone SigLIP inference: person embeddings, attributes, and text queries.

    from siglip_infer import VisionEncoder, TextEncoder

    vision = VisionEncoder()
    result = vision.encode("crop.jpg")
    result.embedding                        # (768,) float32, L2-normalized
    result.attributes["gender"].label       # "Female"
    result.attributes.to_labels()           # {"Gender": "Female", ...}

    text = TextEncoder()
    query = text.encode("người mặc áo đỏ")  # (768,) float32, L2-normalized
    similarity = float(query @ result.embedding)

Both towers emit unit vectors in one shared 768-d space, so cosine similarity is
a plain dot product. Image and text embeddings are only comparable when they come
from the same export — see ``models/siglip_manifest.json``.
"""

from siglip_infer.preprocess import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PIXEL_MEAN,
    PIXEL_STD,
    PreprocessError,
    crop_to_box,
    load_image,
    preprocess_batch,
    preprocess_image,
)
from siglip_infer.runtime import Device, ModelUnavailable
from siglip_infer.taxonomy import (
    AttributeClass,
    AttributeDef,
    AttributeKind,
    AttributePrediction,
    DisplayTier,
    PredictedAttributes,
    Taxonomy,
    TaxonomyError,
    decode_logits,
    load_taxonomy,
)
from siglip_infer.text import (
    CONTEXT_LENGTH,
    SiglipTokenizer,
    TextEncoder,
    canonicalize,
)
from siglip_infer.vision import (
    EMBEDDING_DIM,
    VisionEncoder,
    VisionResult,
    load_manifest,
    verify_manifest_artifacts,
)

__version__ = "2.0.0"

__all__ = [
    "CONTEXT_LENGTH",
    "EMBEDDING_DIM",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "PIXEL_MEAN",
    "PIXEL_STD",
    "AttributeClass",
    "AttributeDef",
    "AttributeKind",
    "AttributePrediction",
    "Device",
    "DisplayTier",
    "ModelUnavailable",
    "PredictedAttributes",
    "PreprocessError",
    "SiglipTokenizer",
    "Taxonomy",
    "TaxonomyError",
    "TextEncoder",
    "VisionEncoder",
    "VisionResult",
    "canonicalize",
    "crop_to_box",
    "decode_logits",
    "load_image",
    "load_manifest",
    "load_taxonomy",
    "preprocess_batch",
    "preprocess_image",
    "verify_manifest_artifacts",
]
