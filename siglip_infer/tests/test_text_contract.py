"""Query canonicalization must match OpenCLIP byte-for-byte.

⚠️ A query cleaned differently still tokenizes, still embeds, and still returns
a ranked result set. It is just answering a slightly different question than the
one the image embeddings were trained against, and no output reveals it.
"""

from __future__ import annotations

import pytest
from siglip_infer.text import canonicalize


def test_punctuation_is_stripped_and_case_is_folded():
    assert canonicalize("A Person, wearing a RED shirt!") == "a person wearing a red shirt"


def test_underscores_become_spaces():
    assert canonicalize("red_shirt") == "red shirt"


def test_whitespace_is_collapsed():
    assert canonicalize("  áo   \n đỏ  ") == "áo đỏ"


def test_vietnamese_diacritics_survive_canonicalization():
    # The vocabulary is multilingual; stripping diacritics would quietly turn
    # every Vietnamese query into a different, worse query.
    assert canonicalize("Người mặc áo đỏ") == "người mặc áo đỏ"


def test_html_entities_are_unescaped_twice_as_openclip_does():
    assert canonicalize("red &amp;amp; blue") == "red blue"


def test_mojibake_is_repaired():
    assert canonicalize("donâ€™t") == "dont"


def test_a_query_of_only_punctuation_canonicalizes_to_nothing():
    # Which is why the tokenizer rejects it: embedding a pad-only sequence
    # returns one arbitrary vector that ranks an arbitrary set of people.
    assert canonicalize("???!!!") == ""


@pytest.mark.parametrize("query", ["a", "người", "x" * 500])
def test_canonicalization_never_returns_leading_or_trailing_space(query):
    result = canonicalize(query)
    assert result == result.strip()
