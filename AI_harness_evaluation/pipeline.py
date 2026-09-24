from __future__ import annotations

import time
from typing import Any

from .contracts import (
    ATTRIBUTE_FIELDS,
    caption_attribute_schema,
    caption_factuality_schema,
    validate_caption_attribute_evaluation,
    validate_caption_factuality,
)
from .dataset import EvaluationTask
from .errors import EvaluationError
from .gemini_client import GeminiJudge
from .prompts import (
    CAPTION_ATTRIBUTE_PROMPT_VERSION,
    CAPTION_FACTUALITY_PROMPT_VERSION,
    caption_attribute_prompt,
    caption_factuality_prompt,
)


def _attribute_summary(checks: dict[str, dict[str, Any]]) -> dict[str, int]:
    summary = {"mentioned": 0, "not_mentioned": 0, "correct": 0, "incorrect": 0, "evaluated": 0}
    for field in ATTRIBUTE_FIELDS:
        check = checks[field]
        if check["mentioned"]:
            summary["mentioned"] += 1
        else:
            summary["not_mentioned"] += 1
        if check["is_correct"] is True:
            summary["correct"] += 1
            summary["evaluated"] += 1
        elif check["is_correct"] is False:
            summary["incorrect"] += 1
            summary["evaluated"] += 1
    return summary


def evaluate_task(task: EvaluationTask, judge: GeminiJudge, *, model: str) -> dict[str, Any]:
    started = time.perf_counter()
    base: dict[str, Any] = {
        "evaluation_schema_version": "3",
        "sample_id": task.sample_id,
        "method": task.method,
        "image_id": task.image_id,
        "filepath": task.relative_image_path,
        "judge_model": model,
        "prompt_versions": {
            "caption_factuality": CAPTION_FACTUALITY_PROMPT_VERSION,
            "caption_attribute": CAPTION_ATTRIBUTE_PROMPT_VERSION,
        },
    }
    try:
        caption_raw = judge.complete(
            prompt=caption_factuality_prompt(caption=task.prediction_caption),
            image=task.image_path.read_bytes(),
            image_path_name=task.image_path.name,
            response_schema=caption_factuality_schema(),
        )
        caption_evaluation = validate_caption_factuality(caption_raw)
        caption_attribute_raw = judge.complete(
            prompt=caption_attribute_prompt(
                caption=task.prediction_caption,
                golden_attributes=task.golden_attributes,
            ),
            image=None,
            image_path_name=None,
            response_schema=caption_attribute_schema(),
        )
        caption_attribute_evaluation = validate_caption_attribute_evaluation(
            caption_attribute_raw
        )
        return {
            **base,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": "success",
            "prediction": {
                "caption": task.prediction_caption,
                "caption_vi": task.prediction_caption_vi,
                "attributes": task.prediction_attributes,
            },
            "golden_attributes": task.golden_attributes,
            "caption_evaluation": caption_evaluation,
            "caption_attribute_evaluation": caption_attribute_evaluation,
            "caption_attribute_summary": _attribute_summary(caption_attribute_evaluation),
        }
    except EvaluationError as exc:
        return {
            **base,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": "error",
            "error": {"code": type(exc).__name__, "message": str(exc)},
        }
    except OSError:
        return {
            **base,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": "error",
            "error": {"code": "image_read", "message": "Query image could not be read"},
        }
    except Exception:
        return {
            **base,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": "error",
            "error": {
                "code": "unexpected",
                "message": "An unexpected local evaluation error occurred",
            },
        }
