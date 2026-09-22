"""Command line for the bundle.

    python -m siglip_infer image crop.jpg
    python -m siglip_infer image crop.jpg --json --embedding-out crop.npy
    python -m siglip_infer text "người mặc áo đỏ" --json
    python -m siglip_infer match crop.jpg "a person in a red shirt"
    python -m siglip_infer taxonomy
    python -m siglip_infer selftest

``image`` and ``text`` are the two capabilities of the bundle; ``match`` scores
one against the other, which is the fastest way to confirm the two towers came
from the same export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from siglip_infer.runtime import Device, ModelUnavailable
from siglip_infer.taxonomy import DisplayTier, PredictedAttributes, load_taxonomy


def _add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=[str(value) for value in Device],
        default=str(Device.AUTO),
        help="auto uses CUDA when available; cuda fails loudly when it is not",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siglip_infer", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("image", help="embed an image and predict its attributes")
    image.add_argument("path", nargs="+", type=Path, help="one or more image files")
    image.add_argument("--model", type=Path, default=None)
    image.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="crop before resizing; fractional (0-1) or pixel coordinates",
    )
    image.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    image.add_argument(
        "--all-attributes",
        action="store_true",
        help="include hidden-tier attributes, which the model predicts no better than a constant",
    )
    image.add_argument(
        "--embedding-out",
        type=Path,
        default=None,
        help="write embeddings to a .npy file (one row per input image)",
    )
    image.add_argument("--batch-size", type=int, default=16)
    _add_device(image)

    text = subparsers.add_parser("text", help="embed a text query")
    text.add_argument("query", nargs="+", help="one or more query strings")
    text.add_argument("--model", type=Path, default=None)
    text.add_argument("--tokenizer", type=Path, default=None)
    text.add_argument("--json", action="store_true")
    text.add_argument("--embedding-out", type=Path, default=None)
    _add_device(text)

    match = subparsers.add_parser("match", help="cosine similarity between an image and queries")
    match.add_argument("path", type=Path)
    match.add_argument("query", nargs="+")
    match.add_argument("--vision-model", type=Path, default=None)
    match.add_argument("--text-model", type=Path, default=None)
    match.add_argument("--tokenizer", type=Path, default=None)
    match.add_argument("--json", action="store_true")
    _add_device(match)

    taxonomy = subparsers.add_parser("taxonomy", help="print the attribute taxonomy")
    taxonomy.add_argument("--json", action="store_true")

    selftest = subparsers.add_parser(
        "selftest", help="load both towers and verify the serving contract end to end"
    )
    _add_device(selftest)

    return parser


def _format_attributes(attributes: PredictedAttributes, *, include_hidden: bool) -> str:
    lines = []
    width = max(len(prediction.attribute.label) for prediction in attributes)
    for prediction in attributes:
        if not include_hidden and prediction.attribute.display_tier is DisplayTier.HIDDEN:
            continue
        name = prediction.attribute.label.ljust(width)
        if prediction.is_multi_label:
            if not prediction.labels:
                value = "—"
            else:
                value = ", ".join(
                    f"{label} ({score:.2f})"
                    for label, score in zip(prediction.labels, prediction.scores, strict=True)
                )
        else:
            value = f"{prediction.label} ({prediction.confidence:.2f})"
            if not prediction.passes_gate(0.35):
                value += "  [low confidence]"
        lines.append(f"  {name}  {value}")
    return "\n".join(lines)


def _command_image(arguments: argparse.Namespace) -> int:
    import numpy as np

    from siglip_infer.vision import DEFAULT_MODEL_PATH, VisionEncoder

    encoder = VisionEncoder(arguments.model or DEFAULT_MODEL_PATH, device=arguments.device)
    boxes = None if arguments.box is None else [arguments.box] * len(arguments.path)
    results = encoder.encode_batch(arguments.path, boxes=boxes, batch_size=arguments.batch_size)

    if arguments.embedding_out is not None:
        np.save(arguments.embedding_out, np.stack([r.embedding for r in results]))

    if arguments.json:
        payload = [
            {"image": str(path), **result.to_dict()}
            for path, result in zip(arguments.path, results, strict=True)
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for path, result in zip(arguments.path, results, strict=True):
        print(f"{path}")
        print(f"  embedding: {result.embedding.shape[0]}-d, unit norm")
        print(_format_attributes(result.attributes, include_hidden=arguments.all_attributes))
        print()
    if arguments.embedding_out is not None:
        print(f"embeddings written to {arguments.embedding_out}")
    return 0


def _command_text(arguments: argparse.Namespace) -> int:
    import numpy as np

    from siglip_infer.text import DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH, TextEncoder

    encoder = TextEncoder(
        arguments.model or DEFAULT_MODEL_PATH,
        tokenizer_path=arguments.tokenizer or DEFAULT_TOKENIZER_PATH,
        device=arguments.device,
    )
    embeddings = encoder.encode_batch(arguments.query)

    if arguments.embedding_out is not None:
        np.save(arguments.embedding_out, embeddings)

    if arguments.json:
        from siglip_infer.text import canonicalize

        print(
            json.dumps(
                [
                    {
                        "query": query,
                        "canonical": canonicalize(query),
                        "embedding_dim": int(vector.shape[0]),
                        "embedding": [float(value) for value in vector],
                    }
                    for query, vector in zip(arguments.query, embeddings, strict=True)
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    for query, vector in zip(arguments.query, embeddings, strict=True):
        preview = ", ".join(f"{value:+.4f}" for value in vector[:6])
        print(f"{query!r}\n  {vector.shape[0]}-d, unit norm: [{preview}, ...]")
    if arguments.embedding_out is not None:
        print(f"embeddings written to {arguments.embedding_out}")
    return 0


def _command_match(arguments: argparse.Namespace) -> int:
    from siglip_infer.text import DEFAULT_MODEL_PATH as TEXT_MODEL
    from siglip_infer.text import DEFAULT_TOKENIZER_PATH, TextEncoder
    from siglip_infer.vision import DEFAULT_MODEL_PATH as VISION_MODEL
    from siglip_infer.vision import VisionEncoder

    vision = VisionEncoder(arguments.vision_model or VISION_MODEL, device=arguments.device)
    text = TextEncoder(
        arguments.text_model or TEXT_MODEL,
        tokenizer_path=arguments.tokenizer or DEFAULT_TOKENIZER_PATH,
        device=arguments.device,
    )
    image_vector = vision.encode(arguments.path).embedding
    scores = text.encode_batch(arguments.query) @ image_vector
    ranked = sorted(
        zip(arguments.query, (float(value) for value in scores), strict=True),
        key=lambda pair: -pair[1],
    )

    if arguments.json:
        print(
            json.dumps(
                {
                    "image": str(arguments.path),
                    "matches": [{"query": q, "cosine": s} for q, s in ranked],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(f"{arguments.path}")
    for query, score in ranked:
        print(f"  {score:+.4f}  {query}")
    return 0


def _command_taxonomy(arguments: argparse.Namespace) -> int:
    taxonomy = load_taxonomy()
    if arguments.json:
        print(json.dumps(_taxonomy_payload(taxonomy), indent=2, ensure_ascii=False))
        return 0
    print(
        f"{len(taxonomy)} attributes, {taxonomy.total_logits} logits, "
        f"multi-label threshold {taxonomy.multilabel_threshold}"
    )
    for attribute in taxonomy:
        kind = "multi" if attribute.is_multi_label else "single"
        print(
            f"\n{attribute.label}  [{attribute.name}] "
            f"({kind}-label, {attribute.display_tier}, logits "
            f"{attribute.start}..{attribute.stop - 1})"
        )
        for entry in attribute.classes:
            print(f"  {entry.logit_index:>3}  {entry.code:<24} {entry.label}")
    return 0


def _taxonomy_payload(taxonomy: Any) -> dict[str, Any]:
    return {
        "num_attributes": len(taxonomy),
        "total_logits": taxonomy.total_logits,
        "multilabel_threshold": taxonomy.multilabel_threshold,
        "attributes": [
            {
                "name": attribute.name,
                "label": attribute.label,
                "type": str(attribute.kind),
                "display_tier": str(attribute.display_tier),
                "logit_start": attribute.start,
                "logit_stop": attribute.stop,
                "classes": [
                    {"logit_index": e.logit_index, "code": e.code, "label": e.label}
                    for e in attribute.classes
                ],
            }
            for attribute in taxonomy
        ],
    }


def _command_selftest(arguments: argparse.Namespace) -> int:
    """Prove the bundle works: real models, real inference, checked outputs."""
    import numpy as np
    from PIL import Image

    from siglip_infer.text import TextEncoder
    from siglip_infer.vision import VisionEncoder, verify_manifest_artifacts

    manifest = verify_manifest_artifacts()
    print(
        f"manifest: {manifest['model']} {manifest['model_version']}, "
        f"opset {manifest['onnx_opset']}, artifact hashes OK"
    )

    taxonomy = load_taxonomy()
    print(f"taxonomy: {len(taxonomy)} attributes over {taxonomy.total_logits} logits  OK")

    vision = VisionEncoder(device=arguments.device)
    print(f"vision:   {vision.providers[0]}")
    generator = np.random.default_rng(0x51_61_4C_49_50)
    synthetic = Image.fromarray(generator.integers(0, 256, size=(256, 96, 3), dtype=np.uint8))
    result = vision.encode(synthetic)
    norm = float(np.linalg.norm(result.embedding))
    assert result.embedding.shape == (768,), result.embedding.shape
    assert abs(norm - 1.0) < 1e-3, norm
    assert len(result.attributes) == len(taxonomy)
    print(f"          768-d unit embedding (norm {norm:.6f}), {len(result.attributes)} attributes")

    text = TextEncoder(device=arguments.device)
    print(f"text:     {text.providers[0]}")
    queries = ["a person wearing a red shirt", "người mặc áo đỏ", "an empty parking lot at night"]
    vectors = text.encode_batch(queries)
    assert vectors.shape == (3, 768), vectors.shape
    cross_lingual = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    print(f"          cosine(en, vi) = {cross_lingual:+.4f}")
    print(f"          cosine(en, unrelated) = {unrelated:+.4f}")
    if cross_lingual <= unrelated:
        print(
            "FAIL: the Vietnamese and English forms of one query should sit closer "
            "than two unrelated queries. The tokenizer or the text tower is wrong.",
            file=sys.stderr,
        )
        return 1

    print("\nselftest passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    handlers = {
        "image": _command_image,
        "text": _command_text,
        "match": _command_match,
        "taxonomy": _command_taxonomy,
        "selftest": _command_selftest,
    }
    try:
        return handlers[arguments.command](arguments)
    except (ModelUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
