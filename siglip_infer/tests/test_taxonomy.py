"""The taxonomy is what gives 200 anonymous numbers meaning.

Every test here names a way the mapping can be wrong while still producing
confident, well-formed, entirely plausible output.
"""

from __future__ import annotations

import json

import pytest
from siglip_infer.taxonomy import (
    AttributeKind,
    DisplayTier,
    TaxonomyError,
    decode_logits,
    load_taxonomy,
)

EXPECTED_ATTRIBUTE_ORDER = (
    "age",
    "gender",
    "body_build",
    "upper_clothing_type",
    "upper_clothing_color",
    "lower_clothing_type",
    "lower_clothing_color",
    "clothing_style",
    "upper_pattern",
    "footwear_type",
    "bag_type",
    "bag_color",
    "headwear",
    "eyewear",
    "face_mask",
    "other_accessories",
    "hair_length",
    "hair_texture",
    "hairstyle",
    "hair_color",
    "carried_objects",
)


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


def test_the_taxonomy_describes_exactly_21_attributes_over_200_logits(taxonomy):
    assert len(taxonomy) == 21
    assert taxonomy.total_logits == 200


def test_attribute_order_is_the_logit_layout_and_must_not_drift(taxonomy):
    # ⚠️ Reordering the taxonomy file silently reinterprets every attribute the
    # model has ever predicted. Nothing errors; the labels just become wrong.
    assert taxonomy.names == EXPECTED_ATTRIBUTE_ORDER


def test_slices_tile_the_logit_vector_without_gaps_or_overlap(taxonomy):
    cursor = 0
    for attribute in taxonomy:
        assert attribute.start == cursor
        assert attribute.stop == cursor + attribute.num_classes
        cursor = attribute.stop
    assert cursor == taxonomy.total_logits


def test_every_logit_index_resolves_to_one_attribute_and_class(taxonomy):
    for index in range(taxonomy.total_logits):
        attribute, entry = taxonomy.describe_logit(index)
        assert entry.logit_index == index
        assert attribute.classes[index - attribute.start] is entry


def test_an_out_of_range_logit_index_is_an_error_not_a_wraparound(taxonomy):
    with pytest.raises(TaxonomyError):
        taxonomy.describe_logit(200)
    with pytest.raises(TaxonomyError):
        taxonomy.describe_logit(-1)


def test_every_class_carries_a_natural_name_distinct_from_its_code(taxonomy):
    for attribute in taxonomy:
        assert attribute.label and attribute.label[0].isupper()
        for entry in attribute.classes:
            assert entry.label, f"{attribute.name}.{entry.code} has no natural name"
            assert "_" not in entry.label, f"{entry.label!r} is a code, not a natural name"


def test_the_four_multi_label_attributes_are_the_ones_a_person_can_have_several_of(taxonomy):
    multi = {a.name for a in taxonomy if a.kind is AttributeKind.MULTI_LABEL}
    assert multi == {"bag_type", "bag_color", "other_accessories", "carried_objects"}


def test_hidden_tier_attributes_are_never_offered_as_filters(taxonomy):
    # A filter *excludes* results. Doing that on an attribute the model predicts
    # no better than a constant silently hides people who match the query.
    for attribute in taxonomy:
        if attribute.display_tier is DisplayTier.HIDDEN:
            assert not attribute.filterable, attribute.name
        if attribute.display_tier is DisplayTier.WEAK:
            assert not attribute.filterable, attribute.name


def test_gender_is_confidence_gated_because_it_has_no_unknown_class(taxonomy):
    # The head must pick male or female even from a crop that cannot support
    # either, so an ungated gender label is the UI stating a fact it cannot know.
    gender = taxonomy["gender"]
    assert "unknown" not in gender.codes
    assert gender.needs_confidence_gate


def test_decoding_returns_one_prediction_per_attribute_in_taxonomy_order(taxonomy):
    predictions = decode_logits([0.0] * 200, taxonomy)
    assert len(predictions) == len(taxonomy)
    assert tuple(p.name for p in predictions) == taxonomy.names


def test_a_single_label_attribute_picks_the_argmax_of_its_own_slice(taxonomy):
    logits = [0.0] * 200
    gender = taxonomy["gender"]
    logits[gender.index_of("female")] = 10.0
    prediction = decode_logits(logits, taxonomy)["gender"]
    assert prediction.code == "female"
    assert prediction.label == "Female"
    assert prediction.confidence > 0.99


def test_a_slice_boundary_error_would_change_the_answer_and_is_detectable(taxonomy):
    # Setting the logit *just before* gender's slice must not affect gender.
    logits = [0.0] * 200
    logits[taxonomy["gender"].start - 1] = 20.0
    prediction = decode_logits(logits, taxonomy)["gender"]
    assert prediction.confidence == pytest.approx(0.5)


def test_multi_label_selects_every_class_over_threshold_not_just_the_best(taxonomy):
    logits = [-20.0] * 200
    bags = taxonomy["bag_type"]
    logits[bags.index_of("backpack")] = 5.0
    logits[bags.index_of("handbag")] = 2.0
    prediction = decode_logits(logits, taxonomy)["bag_type"]
    assert prediction.codes == ("backpack", "handbag")
    assert prediction.labels == ("Backpack", "Handbag")
    assert all(score >= taxonomy.multilabel_threshold for score in prediction.scores)


def test_an_empty_multi_label_set_is_an_answer_not_a_missing_value(taxonomy):
    # "This person carries no bag" and "we never ran inference" are different
    # facts. Decoding produces the first; only an absent result is the second.
    prediction = decode_logits([-20.0] * 200, taxonomy)["bag_type"]
    assert prediction.codes == ()
    assert prediction.to_dict()["codes"] == []


def test_unknown_is_a_real_prediction_a_caller_can_act_on(taxonomy):
    logits = [0.0] * 200
    age = taxonomy["age"]
    logits[age.index_of("unknown")] = 10.0
    prediction = decode_logits(logits, taxonomy)["age"]
    assert prediction.code == "unknown"
    assert prediction.label == "Unknown"


def test_a_wrong_length_logit_vector_is_fatal_rather_than_truncated(taxonomy):
    # Truncating would decode every attribute after the mismatch from the wrong
    # slice, producing a full set of confident, wrong labels.
    with pytest.raises(TaxonomyError, match="got 199"):
        decode_logits([0.0] * 199, taxonomy)
    with pytest.raises(TaxonomyError):
        decode_logits([0.0] * 201, taxonomy)


def test_non_finite_logits_are_rejected_before_they_reach_softmax(taxonomy):
    logits = [0.0] * 200
    logits[3] = float("nan")
    with pytest.raises(TaxonomyError, match="non-finite"):
        decode_logits(logits, taxonomy)


def test_large_magnitude_logits_do_not_overflow_softmax_or_sigmoid(taxonomy):
    for extreme in (1e4, -1e4):
        predictions = decode_logits([extreme] * 200, taxonomy)
        for prediction in predictions:
            if prediction.is_multi_label:
                assert all(0.0 <= score <= 1.0 for score in prediction.scores)
            else:
                assert 0.0 <= prediction.confidence <= 1.0


def test_the_taxonomy_stays_in_sync_with_the_schema_it_was_generated_from():
    # The schema is the model contract; the taxonomy is a derived view of it.
    # If they drift, the natural names describe a layout the model does not use.
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    schema_path = root / "assets" / "attribute_schema.json"
    taxonomy_json = json.loads((root / "assets" / "attribute_taxonomy.json").read_text("utf-8"))
    digest = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    assert taxonomy_json["generated_from"]["sha256"] == digest, (
        "attribute_schema.json changed; re-run tools/build_taxonomy.py"
    )

    schema = json.loads(schema_path.read_text("utf-8"))
    for schema_entry, taxonomy_entry in zip(
        schema["attributes"], taxonomy_json["attributes"], strict=True
    ):
        assert schema_entry["name"] == taxonomy_entry["name"]
        assert schema_entry["type"] == taxonomy_entry["type"]
        assert schema_entry["classes"] == [c["code"] for c in taxonomy_entry["classes"]]


def test_to_labels_is_readable_without_consulting_the_taxonomy(taxonomy):
    labels = decode_logits([0.0] * 200, taxonomy).to_labels()
    assert "Gender" in labels
    assert "Bag type" in labels
    assert isinstance(labels["Bag type"], list)
