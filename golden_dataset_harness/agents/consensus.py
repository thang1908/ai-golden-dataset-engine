"""Caption consensus without a local embedding model.

The remote vLLM service generates several captions. This node groups captions
using Jaccard similarity of their words, then selects the most representative
caption in the largest group. It does not download or run any local ML model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from golden_dataset_harness.schemas.annotation import CaptionCandidate, ConsensusResult
from golden_dataset_harness.schemas.state import PipelineState


def _similarity(left: str, right: str) -> float:
    """Return word-set Jaccard similarity for two captions."""
    left_words = set(re.findall(r"\w+", left.lower()))
    right_words = set(re.findall(r"\w+", right.lower()))
    if not left_words or not right_words:
        return float(left.strip().lower() == right.strip().lower())
    return len(left_words & right_words) / len(left_words | right_words)


def _consensus(candidates: list[CaptionCandidate], threshold: float) -> ConsensusResult:
    if not candidates:
        return ConsensusResult(caption="", agreement_score=0.0, all_candidates=[])
    if len(candidates) == 1:
        return ConsensusResult(
            caption=candidates[0].text,
            agreement_score=1.0,
            all_candidates=candidates,
        )

    similarities = [
        [_similarity(left.text, right.text) for right in candidates]
        for left in candidates
    ]
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for start in range(len(candidates)):
        if start in visited:
            continue
        cluster: list[int] = []
        queue = [start]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            queue.extend(
                index
                for index, score in enumerate(similarities[current])
                if index not in visited and score >= threshold
            )
        clusters.append(cluster)

    best = max(clusters, key=len)
    representative = max(
        best,
        key=lambda index: sum(similarities[index][other] for other in best),
    )
    pairs = [
        similarities[left][right]
        for position, left in enumerate(best)
        for right in best[position + 1 :]
    ]
    return ConsensusResult(
        caption=candidates[representative].text,
        agreement_score=round(sum(pairs) / len(pairs), 4) if pairs else 1.0,
        all_candidates=candidates,
    )


def create_consensus_node(
    similarity_threshold: float = 0.35,
) -> Callable[[PipelineState], Any]:
    """Create a local, dependency-free consensus node."""

    async def node(state: PipelineState) -> dict[str, Any]:
        return {
            "consensus": _consensus(
                state.get("caption_candidates", []),
                similarity_threshold,
            )
        }

    node.__name__ = "consensus"
    return node
