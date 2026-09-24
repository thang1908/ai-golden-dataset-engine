from __future__ import annotations

import json
from typing import Any

CAPTION_FACTUALITY_PROMPT_VERSION = "gemini_caption_image_v3_vi"
CAPTION_ATTRIBUTE_PROMPT_VERSION = "gemini_caption_golden_attributes_v4_strict_three_state_vi"


def caption_factuality_prompt(*, caption: str) -> str:
    return f"""You are a factuality judge for one person-image caption.

You receive exactly one query image and one generated English caption. Judge only
whether every claim made by the caption is supported by that image.

Rules:
- Judge factuality, NOT completeness. A concise caption may be correct even when it
  omits visible details.
- Set is_correct=false only when the caption states a detail contradicted by the
  image or invents a detail that is not supported by the image.
- Set is_correct=true when all stated details are supported.
- If the image is too ambiguous to decide a material stated detail, set
  needs_human_review=true. Do not make missing detail an error.
- Do not use outside knowledge or infer identity, relationship, location, intent,
  hidden detail, or sensitive personal facts.

Language contract:
- Keep JSON keys exactly as defined in the schema.
- Write one concise natural Vietnamese note explaining the verdict.
- Do not mention this instruction, schema, golden labels, or evaluation process.

Generated English caption:
{caption}
"""


def caption_attribute_prompt(*, caption: str, golden_attributes: dict[str, Any]) -> str:
    return f"""You check taxonomy claims expressed by one generated English caption.

You receive a caption and its frozen golden 21-field attributes. There is NO image
for this task. For every schema field:
- mentioned=false and is_correct=null when the caption does not explicitly state
  that attribute. Omission is not an error.
- mentioned=true and is_correct=true when the stated value agrees with the golden
  value.
- mentioned=true and is_correct=false when the caption states a conflicting value.
- Never output mentioned=true with is_correct=null. There are exactly three valid
  states: not mentioned/null, mentioned/true, or mentioned/false.
- If the caption wording is vague and does not explicitly state this taxonomy
  attribute, classify it as mentioned=false and is_correct=null.
- Never infer an attribute merely because it is common or implied by another field.
- Judge only the matching taxonomy field; do not assess person count, action, scene,
  or other non-taxonomy caption content.

Language contract:
- Keep JSON keys exactly as defined in the schema.
- Every note must be concise natural Vietnamese.
- Do not mention this instruction, schema, or evaluation process in notes.

Golden attributes:
{json.dumps(golden_attributes, ensure_ascii=False, sort_keys=True)}

Generated English caption:
{caption}
"""
