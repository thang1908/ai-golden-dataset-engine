"""Consensus engine node.

Clusters caption candidates using sentence-transformer embeddings, selects
the representative caption closest to the cluster centroid, and computes
an agreement score from pairwise cosine similarity.

The SentenceTransformer model is loaded lazily and cached in the closure
created by the factory function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from golden_dataset_harness.schemas.annotation import CaptionCandidate, ConsensusResult
from golden_dataset_harness.schemas.state import PipelineState

logger = logging.getLogger(__name__)


def _compute_consensus(
    candidates: list[CaptionCandidate],
    embeddings: np.ndarray,
    similarity_threshold: float,
) -> ConsensusResult:
    """Core consensus logic.

    1. Compute pairwise cosine similarity matrix.
    2. Build an adjacency graph: edge exists if sim >= threshold.
    3. Find the largest connected component (greedy).
    4. Select the caption closest to the component centroid.
    5. Agreement score = mean pairwise similarity within the component.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(candidates)

    if n == 0:
        return ConsensusResult(
            caption="",
            agreement_score=0.0,
            all_candidates=[],
        )

    if n == 1:
        return ConsensusResult(
            caption=candidates[0].text,
            agreement_score=1.0,
            all_candidates=candidates,
        )

    # Pairwise similarity
    sim_matrix = cosine_similarity(embeddings)

    # Greedy largest-cluster: pick the node with most neighbors, expand
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Find largest cluster via BFS from the best-connected node
    visited: set[int] = set()
    best_cluster: list[int] = []

    for start in range(n):
        if start in visited:
            continue
        cluster: list[int] = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    # Compute centroid of the best cluster
    cluster_embeddings = embeddings[best_cluster]
    centroid = cluster_embeddings.mean(axis=0)

    # Select representative: closest to centroid
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    rep_idx = best_cluster[int(np.argmin(dists))]

    # Agreement score: mean pairwise similarity within the cluster
    cluster_sims: list[float] = []
    for i, ci in enumerate(best_cluster):
        for j, cj in enumerate(best_cluster):
            if i < j:
                cluster_sims.append(float(sim_matrix[ci, cj]))

    agreement = float(np.mean(cluster_sims)) if cluster_sims else 1.0

    return ConsensusResult(
        caption=candidates[rep_idx].text,
        agreement_score=round(agreement, 4),
        all_candidates=candidates,
    )


async def consensus_node(
    state: PipelineState,
    *,
    encoder: Any,  # SentenceTransformer instance
    similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    """Run consensus on caption candidates.

    Args:
        state: Must contain ``caption_candidates``.
        encoder: A SentenceTransformer instance for encoding captions.
        similarity_threshold: Minimum cosine similarity to link two captions.

    Returns:
        Partial state with ``consensus`` (ConsensusResult).
    """
    candidates: list[CaptionCandidate] = state.get("caption_candidates", [])

    if not candidates:
        return {
            "consensus": ConsensusResult(
                caption="", agreement_score=0.0, all_candidates=[]
            ),
            "errors": ["Consensus: no caption candidates"],
        }

    texts = [c.text for c in candidates]
    embeddings = encoder.encode(texts, convert_to_numpy=True)

    result = _compute_consensus(candidates, embeddings, similarity_threshold)
    logger.info(
        "Consensus: selected %r (agreement=%.3f) from %d candidates",
        result.caption[:60],
        result.agreement_score,
        len(candidates),
    )
    return {"consensus": result}


def create_consensus_node(
    embedding_model_name: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.75,
) -> Callable[[PipelineState], Any]:
    """Factory: lazily load the sentence-transformer and bind config.

    The model is loaded on first invocation and cached for subsequent calls.
    """
    _encoder_cache: list[Any] = []  # mutable container for closure caching

    async def node(state: PipelineState) -> dict[str, Any]:
        candidates = state.get("caption_candidates", [])
        if len(candidates) <= 1:
            return {"consensus": _compute_consensus(
                candidates, np.empty((len(candidates), 0)), similarity_threshold
            )}
        if not _encoder_cache:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer: %s", embedding_model_name)
            _encoder_cache.append(SentenceTransformer(embedding_model_name))
        return await consensus_node(
            state, encoder=_encoder_cache[0], similarity_threshold=similarity_threshold
        )

    node.__name__ = "consensus"
    return node
