"""Pipeline evaluation metrics.

Computes aggregate statistics across a batch of AnnotationRecords:
acceptance rates, score distributions, per-attribute breakdowns, and
issue frequency analysis.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from golden_dataset_harness.schemas.annotation import AnnotationRecord


def compute_pipeline_stats(records: list[AnnotationRecord]) -> dict[str, Any]:
    """Compute comprehensive pipeline statistics.

    Args:
        records: List of completed AnnotationRecords.

    Returns:
        Dict with keys: ``total_processed``, ``acceptance_rate``,
        ``mean_confidence``, ``score_distributions``,
        ``per_attribute_distribution``, ``issues_frequency``,
        ``review_status_counts``.
    """
    if not records:
        return {
            "total_processed": 0,
            "acceptance_rate": 0.0,
            "mean_confidence": 0.0,
            "score_distributions": {},
            "per_attribute_distribution": {},
            "issues_frequency": {},
            "review_status_counts": {},
        }

    total = len(records)

    # --- Review status counts ---
    status_counts = Counter(
        r.review_status.value if hasattr(r.review_status, "value") else r.review_status
        for r in records
    )

    accepted = status_counts.get("auto_accepted", 0) + status_counts.get("human_approved", 0)
    acceptance_rate = accepted / total

    # --- Score distributions ---
    confidences = [r.confidence for r in records]
    judge_scores = [r.judge_score for r in records]
    consensus_scores = [r.consensus_score for r in records]
    grounding_scores = [r.grounding_score for r in records]

    def _dist(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "median": round(float(np.median(arr)), 4),
        }

    score_distributions = {
        "confidence": _dist(confidences),
        "judge_score": _dist(judge_scores),
        "consensus_score": _dist(consensus_scores),
        "grounding_score": _dist(grounding_scores),
    }

    # --- Per-attribute distribution ---
    attr_counter: dict[str, Counter] = {}
    for record in records:
        attr_dict = record.attributes.model_dump()
        for attr_name, value in attr_dict.items():
            if attr_name not in attr_counter:
                attr_counter[attr_name] = Counter()
            attr_counter[attr_name][value] += 1

    per_attribute = {
        attr: dict(counter.most_common())
        for attr, counter in attr_counter.items()
    }

    # --- Issues frequency ---
    issue_counter: Counter = Counter()
    for record in records:
        for issue in record.issues:
            issue_counter[issue] += 1

    return {
        "total_processed": total,
        "acceptance_rate": round(acceptance_rate, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "score_distributions": score_distributions,
        "per_attribute_distribution": per_attribute,
        "issues_frequency": dict(issue_counter.most_common()),
        "review_status_counts": dict(status_counts),
    }
