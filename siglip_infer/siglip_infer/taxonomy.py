"""The attribute taxonomy: what the 200 logits mean.

The model emits one flat vector of 200 numbers. This module is what turns that
into named attributes with natural class names. It reads
``assets/attribute_taxonomy.json`` and nothing else — no torch, no network, no
model file — so it is importable and testable on a machine that has none.

Two rules the rest of the bundle depends on:

* **Slices are positional.** Attribute order in the taxonomy file defines the
  layout of the logit vector. Reordering the file silently reinterprets every
  attribute, which is why the slices are validated against ``total_logits`` on
  load rather than trusted.
* **Three states are distinct.** ``None`` means inference never ran.
  ``"unknown"`` means the model ran and judged the attribute unclear — an
  answer, not an absence. An empty multi-label set means the model ran and found
  none. Collapsing any two of these loses information the product depends on.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "assets" / "attribute_taxonomy.json"

#: The taxonomy version this code understands. A different version in the file
#: is a hard error: a silently reinterpreted logit layout is worse than a crash.
SUPPORTED_TAXONOMY_VERSION = 1

#: Pinned so a truncated or extended taxonomy cannot quietly disagree with the
#: exported ONNX graph, which emits exactly this many logits.
EXPECTED_TOTAL_LOGITS = 200
EXPECTED_ATTRIBUTES = 21


class TaxonomyError(ValueError):
    """The taxonomy file is unusable.

    Always fatal. A malformed taxonomy means every decoded attribute is of
    unknown meaning, so there is no degraded mode worth offering.
    """


class AttributeKind(StrEnum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


class DisplayTier(StrEnum):
    """What a UI is allowed to do with an attribute.

    Derived from measured lift over an always-predict-the-majority baseline,
    not from raw accuracy.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNVALIDATED = "unvalidated"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class AttributeClass:
    """One class of one attribute, and where it sits in the logit vector."""

    index: int
    """Position within the attribute's own slice."""
    logit_index: int
    """Absolute position in the flat 200-logit vector."""
    code: str
    """The stable machine identifier, e.g. ``tshirt_short_sleeve``."""
    label: str
    """The natural, human-readable name, e.g. ``T-shirt (short sleeve)``."""


@dataclass(frozen=True, slots=True)
class AttributeDef:
    """One attribute and its slice of the logit vector."""

    name: str
    label: str
    kind: AttributeKind
    display_tier: DisplayTier
    filterable: bool
    needs_confidence_gate: bool
    start: int
    """Inclusive start index into the flat logit vector."""
    stop: int
    """Exclusive stop index into the flat logit vector."""
    classes: tuple[AttributeClass, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def is_multi_label(self) -> bool:
        return self.kind is AttributeKind.MULTI_LABEL

    @property
    def displayable(self) -> bool:
        return self.display_tier is not DisplayTier.HIDDEN

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(entry.code for entry in self.classes)

    def label_of(self, code: str) -> str:
        """Natural class name for ``code``."""
        return self._class(code).label

    def index_of(self, code: str) -> int:
        """Absolute index of ``code`` within the flat 200-logit vector."""
        return self._class(code).logit_index

    def _class(self, code: str) -> AttributeClass:
        for entry in self.classes:
            if entry.code == code:
                return entry
        raise TaxonomyError(f"{self.name!r} has no class {code!r}")


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """Every attribute, in logit order."""

    version: int
    total_logits: int
    multilabel_threshold: float
    attributes: tuple[AttributeDef, ...]

    def __iter__(self):
        return iter(self.attributes)

    def __len__(self) -> int:
        return len(self.attributes)

    def __contains__(self, name: object) -> bool:
        return any(attribute.name == name for attribute in self.attributes)

    def __getitem__(self, name: str) -> AttributeDef:
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        raise TaxonomyError(f"unknown attribute {name!r}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(attribute.name for attribute in self.attributes)

    def describe_logit(self, index: int) -> tuple[AttributeDef, AttributeClass]:
        """Which attribute and class does flat logit ``index`` belong to?"""
        for attribute in self.attributes:
            if attribute.start <= index < attribute.stop:
                return attribute, attribute.classes[index - attribute.start]
        raise TaxonomyError(f"logit index {index} is outside 0..{self.total_logits - 1}")


def _parse(payload: dict[str, Any]) -> Taxonomy:
    version = payload.get("taxonomy_version")
    if version != SUPPORTED_TAXONOMY_VERSION:
        raise TaxonomyError(
            f"taxonomy version {version!r} is not supported (expected {SUPPORTED_TAXONOMY_VERSION})"
        )

    raw = payload.get("attributes")
    if not isinstance(raw, list) or not raw:
        raise TaxonomyError("taxonomy has no attributes")

    attributes: list[AttributeDef] = []
    cursor = 0
    for entry in raw:
        name = entry["name"]
        try:
            kind = AttributeKind(entry["type"])
        except ValueError:
            raise TaxonomyError(f"attribute {name!r} has unknown type {entry['type']!r}") from None
        try:
            tier = DisplayTier(entry["display_tier"])
        except ValueError:
            raise TaxonomyError(
                f"attribute {name!r} has unknown display tier {entry['display_tier']!r}"
            ) from None

        classes = tuple(
            AttributeClass(
                index=item["index"],
                logit_index=item["logit_index"],
                code=item["code"],
                label=item["label"],
            )
            for item in entry["classes"]
        )
        if len(classes) != entry["num_classes"]:
            raise TaxonomyError(
                f"attribute {name!r} declares {entry['num_classes']} classes "
                f"but lists {len(classes)}"
            )
        if len({item.code for item in classes}) != len(classes):
            raise TaxonomyError(f"attribute {name!r} has duplicate class codes")
        # ⚠️ A class whose logit_index disagrees with its position still decodes
        # to a plausible label -- just the wrong one, for every crop, forever.
        for offset, item in enumerate(classes):
            if item.index != offset or item.logit_index != cursor + offset:
                raise TaxonomyError(
                    f"attribute {name!r} class {item.code!r} claims logit "
                    f"{item.logit_index} at position {item.index}; "
                    f"its slice puts it at {cursor + offset}"
                )
        if entry["logit_start"] != cursor or entry["logit_stop"] != cursor + len(classes):
            raise TaxonomyError(
                f"attribute {name!r} declares slice "
                f"[{entry['logit_start']}, {entry['logit_stop']}) "
                f"but follows its predecessors at [{cursor}, {cursor + len(classes)})"
            )

        attributes.append(
            AttributeDef(
                name=name,
                label=entry["label"],
                kind=kind,
                display_tier=tier,
                filterable=bool(entry["filterable"]),
                needs_confidence_gate=bool(entry["needs_confidence_gate"]),
                start=cursor,
                stop=cursor + len(classes),
                classes=classes,
            )
        )
        cursor += len(classes)

    if len({attribute.name for attribute in attributes}) != len(attributes):
        raise TaxonomyError("taxonomy has duplicate attribute names")
    if cursor != payload.get("total_logits"):
        raise TaxonomyError(
            f"attribute slices sum to {cursor} logits but the taxonomy declares "
            f"{payload.get('total_logits')}"
        )
    if cursor != EXPECTED_TOTAL_LOGITS:
        raise TaxonomyError(
            f"taxonomy totals {cursor} logits but this build expects "
            f"{EXPECTED_TOTAL_LOGITS}; the taxonomy and the exported model disagree"
        )
    if len(attributes) != EXPECTED_ATTRIBUTES:
        raise TaxonomyError(
            f"taxonomy has {len(attributes)} attributes, expected {EXPECTED_ATTRIBUTES}"
        )

    threshold = float(payload["multilabel_threshold"])
    if not 0.0 < threshold < 1.0:
        raise TaxonomyError(f"multilabel_threshold {threshold} is not a probability")

    return Taxonomy(
        version=version,
        total_logits=cursor,
        multilabel_threshold=threshold,
        attributes=tuple(attributes),
    )


@lru_cache(maxsize=4)
def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load the taxonomy, cached for the process lifetime.

    It is immutable and fixed for the deployment, so a single parse is correct.
    Pass ``path`` in tests to load a variant.
    """
    source = Path(path) if path is not None else TAXONOMY_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TaxonomyError(f"attribute taxonomy not found at {source}") from None
    except json.JSONDecodeError as error:
        raise TaxonomyError(f"attribute taxonomy at {source} is not valid JSON: {error}") from error
    return _parse(payload)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def _softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax over one short slice."""
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sigmoid(value: float) -> float:
    # Branch on sign to keep exp() away from overflow for large-magnitude logits.
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class AttributePrediction:
    """One attribute's decoded value.

    Single-label: ``code``/``label`` hold the winning class and ``confidence``
    its softmax probability. Multi-label: ``codes``/``labels`` hold every class
    at or above the threshold and ``scores`` their individual sigmoid
    probabilities, in taxonomy order.
    """

    attribute: AttributeDef
    code: str | None = None
    label: str | None = None
    confidence: float = 0.0
    codes: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()

    @property
    def name(self) -> str:
        return self.attribute.name

    @property
    def is_multi_label(self) -> bool:
        return self.attribute.is_multi_label

    def passes_gate(self, threshold: float) -> bool:
        """Whether a confidence-gated attribute may be shown at ``threshold``.

        Only meaningful for single-label attributes; multi-label membership is
        already a per-class threshold decision.
        """
        if self.is_multi_label or not self.attribute.needs_confidence_gate:
            return True
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        common = {
            "attribute": self.attribute.name,
            "attribute_label": self.attribute.label,
            "type": str(self.attribute.kind),
            "display_tier": str(self.attribute.display_tier),
        }
        if self.is_multi_label:
            return {
                **common,
                "codes": list(self.codes),
                "labels": list(self.labels),
                "scores": [round(score, 6) for score in self.scores],
            }
        return {
            **common,
            "code": self.code,
            "label": self.label,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True, slots=True)
class PredictedAttributes:
    """Every attribute decoded from one logit vector, in taxonomy order."""

    predictions: tuple[AttributePrediction, ...]

    def __iter__(self):
        return iter(self.predictions)

    def __len__(self) -> int:
        return len(self.predictions)

    def __contains__(self, name: object) -> bool:
        return any(prediction.name == name for prediction in self.predictions)

    def __getitem__(self, name: str) -> AttributePrediction:
        for prediction in self.predictions:
            if prediction.name == name:
                return prediction
        raise TaxonomyError(f"unknown attribute {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """``{attribute_name: decoded value}``, keyed for machine consumption."""
        return {prediction.name: prediction.to_dict() for prediction in self.predictions}

    def to_labels(self) -> dict[str, str | list[str]]:
        """``{natural attribute name: natural class name(s)}``, for humans."""
        return {
            prediction.attribute.label: (
                list(prediction.labels) if prediction.is_multi_label else (prediction.label or "")
            )
            for prediction in self.predictions
        }


def decode_logits(
    logits: Sequence[float],
    taxonomy: Taxonomy | None = None,
    *,
    multilabel_threshold: float | None = None,
) -> PredictedAttributes:
    """Decode one flat attribute logit vector into named, labelled values.

    ``logits`` must be exactly ``taxonomy.total_logits`` long. A length mismatch
    is fatal rather than tolerated: it means the model and the taxonomy disagree,
    and every attribute after the first mismatched slice would be silently wrong.
    """
    taxonomy = taxonomy or load_taxonomy()
    values = [float(value) for value in logits]
    if len(values) != taxonomy.total_logits:
        raise TaxonomyError(
            f"expected {taxonomy.total_logits} logits, got {len(values)}; "
            "the model and the attribute taxonomy disagree"
        )
    if any(not math.isfinite(value) for value in values):
        raise TaxonomyError("attribute logits contain non-finite values")

    threshold = (
        taxonomy.multilabel_threshold if multilabel_threshold is None else multilabel_threshold
    )

    predictions: list[AttributePrediction] = []
    for attribute in taxonomy:
        window = values[attribute.start : attribute.stop]

        if attribute.kind is AttributeKind.SINGLE_LABEL:
            probabilities = _softmax(window)
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            winner = attribute.classes[best]
            predictions.append(
                AttributePrediction(
                    attribute=attribute,
                    code=winner.code,
                    label=winner.label,
                    confidence=probabilities[best],
                )
            )
            continue

        codes: list[str] = []
        labels: list[str] = []
        scores: list[float] = []
        for entry, logit in zip(attribute.classes, window, strict=True):
            probability = _sigmoid(logit)
            if probability >= threshold:
                codes.append(entry.code)
                labels.append(entry.label)
                scores.append(probability)
        predictions.append(
            AttributePrediction(
                attribute=attribute,
                codes=tuple(codes),
                labels=tuple(labels),
                scores=tuple(scores),
            )
        )

    return PredictedAttributes(predictions=tuple(predictions))
