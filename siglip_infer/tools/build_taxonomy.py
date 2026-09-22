#!/usr/bin/env python3
"""Generate ``assets/attribute_taxonomy.json`` from ``assets/attribute_schema.json``.

The schema file is the *contract*: it fixes which slice of the 200-logit vector
belongs to which attribute, and in which order. This tool adds the two things a
consumer needs on top of that and nothing else:

* **natural class names** — ``tshirt_short_sleeve`` -> ``T-shirt (short sleeve)``,
* **display tiers** — measured predictive lift, which decides what a UI may show
  prominently and what it may filter on.

Run it only after changing the schema or a label; the generated file is checked
in so the runtime never needs this tool, torch, or a network.

    python tools/build_taxonomy.py

⚠️ Attribute order in the schema *is* the logit layout. Reordering the schema
silently reinterprets every attribute a stored model ever predicted, so this
tool refuses to emit a taxonomy whose slices do not sum to ``total_logits``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = BUNDLE_ROOT / "assets" / "attribute_schema.json"
TAXONOMY_PATH = BUNDLE_ROOT / "assets" / "attribute_taxonomy.json"

TAXONOMY_VERSION = 1
EXPECTED_TOTAL_LOGITS = 200
EXPECTED_ATTRIBUTES = 21

#: Decision threshold for multi-label attributes. Measured, not conventional:
#: 0.20 maximized binary macro F1 for this head on the private human-labelled
#: evaluation set. It belongs to the deployed MLP-512 head, not to the taxonomy.
MULTILABEL_THRESHOLD = 0.20

#: What a UI may do with each attribute, from measured lift over an
#: always-predict-the-majority-class baseline — *not* from raw accuracy.
#: Two attributes have the highest raw accuracy in the whole set and carry
#: almost no information; a UI that ranks by accuracy leads with exactly those.
#: Attributes missing from this table default to ``unvalidated``.
DISPLAY_TIERS: dict[str, str] = {
    # lift >= 0.20 — show prominently, safe as a search facet
    "upper_clothing_color": "strong",
    "gender": "strong",
    "upper_clothing_type": "strong",
    "lower_clothing_type": "strong",
    "hair_length": "strong",
    "hairstyle": "strong",
    "lower_clothing_color": "strong",
    # lift 0.05 - 0.20 — show, but gate on confidence
    "footwear_type": "moderate",
    "upper_pattern": "moderate",
    "eyewear": "moderate",
    "headwear": "moderate",
    # weak at top-1: age recovers at top-2, face mask has only small lift
    "age": "weak",
    "face_mask": "weak",
    # at or below the majority prior — the model adds nothing over a constant
    "hair_color": "hidden",
    "hair_texture": "hidden",
    "clothing_style": "hidden",
    "body_build": "hidden",
    # multi-label: per-class reliability unmeasured
    "bag_type": "unvalidated",
    "bag_color": "unvalidated",
    "other_accessories": "unvalidated",
    "carried_objects": "unvalidated",
}

#: Tiers a UI may offer as a *filter*. Excluding on a value is a stronger claim
#: than displaying it, so weak and hidden attributes are display-only.
FILTERABLE_TIERS = frozenset({"strong", "moderate", "unvalidated"})

#: Attributes with no ``unknown`` class: the head must pick a value even from a
#: crop that cannot support one, so a low-confidence prediction here is the
#: model guessing, not the model knowing.
NO_UNKNOWN_CLASS = frozenset({"gender"})

ATTRIBUTE_LABELS: dict[str, str] = {
    "age": "Age",
    "gender": "Gender",
    "body_build": "Body build",
    "upper_clothing_type": "Upper clothing type",
    "upper_clothing_color": "Upper clothing color",
    "lower_clothing_type": "Lower clothing type",
    "lower_clothing_color": "Lower clothing color",
    "clothing_style": "Clothing style",
    "upper_pattern": "Upper pattern",
    "footwear_type": "Footwear type",
    "bag_type": "Bag type",
    "bag_color": "Bag color",
    "headwear": "Headwear",
    "eyewear": "Eyewear",
    "face_mask": "Face mask",
    "other_accessories": "Other accessories",
    "hair_length": "Hair length",
    "hair_texture": "Hair texture",
    "hairstyle": "Hairstyle",
    "hair_color": "Hair color",
    "carried_objects": "Carried objects",
}

#: Class codes whose natural name is not just the code with spaces. Anything
#: absent falls through to ``_humanize`` — which is correct for most codes and
#: wrong in a visible, fixable way for the rest.
CLASS_LABELS: dict[str, str] = {
    # age
    "young_adult": "Young adult",
    "middle_aged": "Middle-aged",
    # upper clothing
    "tshirt_short_sleeve": "T-shirt (short sleeve)",
    "tshirt_long_sleeve": "T-shirt (long sleeve)",
    "polo_short_sleeve": "Polo shirt (short sleeve)",
    "polo_long_sleeve": "Polo shirt (long sleeve)",
    "shirt_short_sleeve": "Shirt (short sleeve)",
    "shirt_long_sleeve": "Shirt (long sleeve)",
    "sun_protective_jacket": "Sun-protective jacket",
    "one_piece_dress": "One-piece dress",
    "traditional": "Traditional dress",
    # colors
    "beige_cream": "Beige / cream",
    "gray_white": "Gray / white",
    "partially_gray": "Partially gray",
    # lower clothing
    "dress_trousers": "Dress trousers",
    "cargo_pants": "Cargo pants",
    "regular_shorts": "Shorts (regular)",
    "knee_length_shorts": "Shorts (knee-length)",
    "tailored_shorts": "Shorts (tailored)",
    "skirt_short": "Skirt (short)",
    "skirt_knee_length": "Skirt (knee-length)",
    "skirt_long": "Skirt (long)",
    # style
    "formal_style": "Formal",
    "casual_style": "Casual",
    "workwear_style": "Workwear",
    "sportswear_style": "Sportswear",
    "traditional_style": "Traditional",
    # pattern
    "polka_dot": "Polka dot",
    # footwear
    "casual_sneakers": "Sneakers (casual)",
    "athletic_shoes": "Athletic shoes",
    "dress_shoes": "Dress shoes",
    "flip_flops": "Flip-flops",
    "high_heels": "High heels",
    "roller_skates": "Roller skates",
    # bags
    "crossbody_bag": "Crossbody bag",
    "shoulder_bag": "Shoulder bag",
    "tote_bag": "Tote bag",
    "shopping_bag": "Shopping bag",
    "plastic_bag": "Plastic bag",
    "waist_bag": "Waist bag",
    "duffel_bag": "Duffel bag",
    # headwear
    "baseball_cap": "Baseball cap",
    "brimmed_hat": "Brimmed hat",
    "conical_hat": "Conical hat (nón lá)",
    # face
    "cloth_mask": "Cloth mask",
    "medical_mask": "Medical mask",
    "full_face_covering": "Full face covering",
    # hair
    "short_hair": "Short",
    "medium_hair": "Medium",
    "long_hair": "Long",
    "straight_hair": "Straight",
    "curly_hair": "Curly",
    "loose_hair": "Loose",
    "braided_hair": "Braided",
    # carried objects
    "baby_stroller": "Baby stroller",
    "shopping_cart": "Shopping cart",
    "sport_racket": "Sport racket",
    "handheld_fan": "Handheld fan",
    # generic
    "none": "None",
    "other": "Other",
    "unknown": "Unknown",
}

#: Attributes whose class names read better in context than in isolation.
#: ``hair_length: short`` is clear; a bare "Short" in a chip list is not.
QUALIFIED_LABELS: dict[str, str] = {
    "hair_length": "{label} hair",
    "hair_texture": "{label} hair",
    "hairstyle": "{label} hair",
    "clothing_style": "{label} style",
}

#: Attributes that qualify their classes but have codes where the qualifier
#: would read wrong ("Unknown hair", "Balding hair", "Shaved hair").
QUALIFIER_EXEMPT: dict[str, frozenset[str]] = {
    "hair_length": frozenset({"unknown", "shaved", "balding"}),
    "hair_texture": frozenset({"unknown"}),
    "hairstyle": frozenset({"unknown", "bun", "ponytail"}),
    "clothing_style": frozenset(),
}


def _humanize(code: str) -> str:
    """``sports_jersey`` -> ``Sports jersey``. The fallback, not the rule."""
    spaced = code.replace("_", " ").strip()
    return spaced[:1].upper() + spaced[1:] if spaced else spaced


def _class_label(attribute: str, code: str) -> str:
    label = CLASS_LABELS.get(code, _humanize(code))
    pattern = QUALIFIED_LABELS.get(attribute)
    if pattern is not None and code not in QUALIFIER_EXEMPT.get(attribute, frozenset()):
        return pattern.format(label=label[:1].lower() + label[1:]).capitalize()
    return label


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(schema_path: Path) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("version") != 1:
        raise SystemExit(f"unsupported attribute schema version {schema.get('version')!r}")

    attributes: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    cursor = 0
    for entry in schema["attributes"]:
        name = entry["name"]
        classes = list(entry["classes"])
        if len(classes) != entry["num_classes"]:
            raise SystemExit(f"attribute {name!r} lists {len(classes)} of {entry['num_classes']}")
        if len(set(classes)) != len(classes):
            raise SystemExit(f"attribute {name!r} has duplicate class codes")

        tier = DISPLAY_TIERS.get(name, "unvalidated")
        attribute_label = ATTRIBUTE_LABELS.get(name, _humanize(name))
        class_entries = []
        for offset, code in enumerate(classes):
            label = _class_label(name, code)
            class_entries.append(
                {
                    "index": offset,
                    "logit_index": cursor + offset,
                    "code": code,
                    "label": label,
                }
            )
            flat.append(
                {
                    "logit_index": cursor + offset,
                    "attribute": name,
                    "attribute_label": attribute_label,
                    "class_index": offset,
                    "class_code": code,
                    "class_label": label,
                }
            )

        attributes.append(
            {
                "name": name,
                "label": attribute_label,
                "type": entry["type"],
                "display_tier": tier,
                "filterable": tier in FILTERABLE_TIERS,
                "needs_confidence_gate": name in NO_UNKNOWN_CLASS or tier == "moderate",
                "logit_start": cursor,
                "logit_stop": cursor + len(classes),
                "num_classes": len(classes),
                "classes": class_entries,
            }
        )
        cursor += len(classes)

    if cursor != schema["total_logits"] or cursor != EXPECTED_TOTAL_LOGITS:
        raise SystemExit(
            f"slices sum to {cursor} logits; schema says {schema['total_logits']}, "
            f"this build expects {EXPECTED_TOTAL_LOGITS}"
        )
    if len(attributes) != EXPECTED_ATTRIBUTES:
        raise SystemExit(f"expected {EXPECTED_ATTRIBUTES} attributes, built {len(attributes)}")

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "schema_version": schema["version"],
        "generated_from": {
            "file": schema_path.name,
            "sha256": _sha256(schema_path),
        },
        "num_attributes": len(attributes),
        "total_logits": cursor,
        "multilabel_threshold": MULTILABEL_THRESHOLD,
        "decode_rules": {
            "single_label": "softmax over the attribute's slice, then argmax",
            "multi_label": (
                "independent sigmoid per class; every class at or above "
                "multilabel_threshold is selected, and an empty set is a valid answer "
                "meaning 'the model ran and found none'"
            ),
            "unknown": (
                "'unknown' is an answer -- the model ran and judged the attribute "
                "unclear. It is not the same as inference never having run."
            ),
        },
        "display_tiers": {
            "strong": "Display prominently; safe to expose as a search facet.",
            "moderate": (
                "Display, confidence-gated. A positive is more informative than a negative."
            ),
            "weak": "Supporting detail only. Never a filter.",
            "unvalidated": "Per-class reliability unmeasured. Display without prominence.",
            "hidden": "Do not display or filter -- the model adds nothing over a constant.",
        },
        "attributes": attributes,
        "index_to_class": flat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--out", type=Path, default=TAXONOMY_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the generated taxonomy differs from --out",
    )
    arguments = parser.parse_args()

    payload = json.dumps(build(arguments.schema), indent=2, ensure_ascii=False) + "\n"
    if arguments.check:
        current = arguments.out.read_text(encoding="utf-8") if arguments.out.is_file() else ""
        if current != payload:
            raise SystemExit(f"{arguments.out} is stale; re-run tools/build_taxonomy.py")
        print(f"{arguments.out} is up to date")
        return
    arguments.out.write_text(payload, encoding="utf-8")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
