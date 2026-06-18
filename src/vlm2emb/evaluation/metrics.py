"""Evaluation metrics for retrieval and classification.

Provides ranking metrics for retrieval and classification evaluation.

Supported metrics:
- Hit@K
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vlm2emb.utils.logging import RankLogger

logger = RankLogger(__name__)


# ============================================================================
# Metric classes.
# ============================================================================

SUPPORTED_RANKING_METRICS = frozenset({"hit", "mrr", "map", "ndcg"})
DEFAULT_RETRIEVAL_METRICS: tuple[str, ...] = (
    "hit@1",
    "hit@5",
    "hit@10",
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "mrr@1",
    "mrr@5",
    "mrr@10",
    "map@1",
    "map@5",
    "map@10",
)


@dataclass(frozen=True)
class RankingMetricSpec:
    """Parsed ranking metric name."""

    name: str
    family: str
    k: int


def parse_ranking_metric_names(
    metrics: Sequence[str] | None = None,
) -> tuple[RankingMetricSpec, ...]:
    """Parse and validate complete ranking metric names."""
    metric_names = DEFAULT_RETRIEVAL_METRICS if metrics is None else tuple(metrics)
    if not metric_names:
        raise ValueError("metrics must not be empty; expected names like 'hit@1'")

    parsed: list[RankingMetricSpec] = []
    seen: set[str] = set()
    for metric_name in metric_names:
        if not isinstance(metric_name, str):
            raise ValueError(
                f"Invalid metric name {metric_name!r}; expected '<metric>@<k>'"
            )
        if "@" not in metric_name:
            raise ValueError(
                f"Invalid metric name {metric_name!r}; expected '<metric>@<k>'"
            )

        family, k_text = metric_name.rsplit("@", 1)
        if family not in SUPPORTED_RANKING_METRICS:
            raise ValueError(
                f"Unknown ranking metric family {family!r} in {metric_name!r}; "
                f"supported metrics: {sorted(SUPPORTED_RANKING_METRICS)}"
            )
        try:
            k = int(k_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid metric name {metric_name!r}; k must be a positive integer"
            ) from exc
        if k <= 0:
            raise ValueError(
                f"Invalid metric name {metric_name!r}; k must be a positive integer"
            )

        canonical_name = f"{family}@{k}"
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        parsed.append(RankingMetricSpec(name=canonical_name, family=family, k=k))

    return tuple(parsed)


class RankingMetrics:
    """GPU-accelerated batch ranking metrics for retrieval evaluation.

    All metrics are computed via batch tensor operations on GPU.
    Only ``.item()`` at the end converts results to Python float.

    Args:
        metrics: Complete metric names to compute, e.g. ("hit@1", "ndcg@5").
    """

    def __init__(self, metrics: Sequence[str] | None = None):
        self.metric_specs = parse_ranking_metric_names(metrics)
        self.metrics = tuple(spec.name for spec in self.metric_specs)
        self.max_k = max(spec.k for spec in self.metric_specs)

    def compute(
        self,
        relevance: torch.Tensor,
        n_relevant: torch.Tensor,
        all_rel_scores: torch.Tensor,
    ) -> dict[str, float]:
        """Compute requested metrics from GPU tensor inputs.

        Args:
            relevance: [N_query, max_k] relevance scores at top-K positions.
                       Binary (0/1) or graded (float scores).
            n_relevant: [N_query] total relevant docs per query.
            all_rel_scores: [N_query, max_n_rel] all relevance scores
                            for ideal DCG computation.
        Returns:
            dict[str, float] of metric scores, e.g. {"hit@1": 0.85, ...}
        """
        if relevance.ndim != 2:
            raise ValueError(f"relevance must be a 2D tensor, got shape {tuple(relevance.shape)}")
        device = relevance.device
        max_k = relevance.shape[1]
        if max_k < self.max_k:
            raise ValueError(
                f"relevance width ({max_k}) is smaller than requested max_k={self.max_k}"
            )

        empty_gt_count = (n_relevant == 0).sum().item()
        if empty_gt_count > 0:
            logger.warning(
                f"{empty_gt_count} queries have empty ground truth "
                f"(n_relevant=0), contributing 0.0 to all metrics"
            )

        # Non-NDCG metrics use relevance as membership. NDCG uses graded scores.
        binary_rel = (relevance > 0).float()
        positions = torch.arange(1, max_k + 1, device=device, dtype=torch.float32)
        n_rel = n_relevant.float().clamp(min=1)
        ideal_rel = None
        if any(spec.family == "ndcg" for spec in self.metric_specs):
            ideal_rel, _ = all_rel_scores.sort(dim=1, descending=True)

        scores: dict[str, float] = {}
        binary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for spec in self.metric_specs:
            k = spec.k
            effective_k = min(k, max_k)

            if spec.family == "ndcg":
                rel_k = relevance[:, :effective_k].float()
                discount = 1.0 / torch.log2(positions[:effective_k] + 1)
                dcg = (rel_k * discount).sum(dim=1)
                if ideal_rel is None:
                    raise RuntimeError("ideal_rel cache was not initialized")
                n_ideal = min(effective_k, ideal_rel.shape[1])
                ideal_k = ideal_rel[:, :n_ideal]
                idcg = (ideal_k * discount[:n_ideal]).sum(dim=1).clamp(min=1e-10)
                scores[spec.name] = (dcg / idcg).mean().item()
                continue

            if effective_k not in binary_cache:
                bin_k = binary_rel[:, :effective_k]
                hits = bin_k.sum(dim=1)
                binary_cache[effective_k] = (bin_k, hits)
            else:
                bin_k, hits = binary_cache[effective_k]

            if spec.family == "hit":
                scores[spec.name] = (hits > 0).float().mean().item()
            elif spec.family == "mrr":
                has_hit = hits > 0
                if effective_k == 0:
                    rr = torch.zeros_like(hits)
                else:
                    first_hit_pos = bin_k.argmax(dim=1) + 1
                    rr = torch.where(
                        has_hit,
                        1.0 / first_hit_pos.float(),
                        torch.zeros_like(hits),
                    )
                scores[spec.name] = rr.mean().item()
            elif spec.family == "map":
                cumhits = bin_k.cumsum(dim=1)
                precision_at_pos = cumhits / positions[:effective_k]
                ap_denom = n_rel.clamp(max=float(k))
                ap = (precision_at_pos * bin_k).sum(dim=1) / ap_denom
                scores[spec.name] = ap.mean().item()

        return scores


# ============================================================================
# Convenience functions.
# ============================================================================

def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    rel_cand_ids: list[list[str]],
    candidate_ids: list[str],
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics from embeddings (compatibility adapter).

    Converts numpy inputs to GPU tensors internally, delegates to
    ``RankingMetrics.compute``, and returns ``dict[str, float]``.

    Args:
        query_embeddings: Query embeddings [N_query, D]
        candidate_embeddings: Candidate embeddings [N_cand, D]
        rel_cand_ids: List of relevant candidate IDs for each query
        candidate_ids: List of candidate IDs [N_cand]
        metrics: Complete metric names to compute

    Returns:
        Dictionary of metric scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_computer = RankingMetrics(metrics=metrics)
    max_k = metric_computer.max_k

    # Convert to tensors
    q_emb = torch.as_tensor(query_embeddings, dtype=torch.float32, device=device)
    c_emb = torch.as_tensor(candidate_embeddings, dtype=torch.float32, device=device)

    # Build candidate id to index mapping
    cand_id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    n_query = q_emb.shape[0]
    n_cand = c_emb.shape[0]
    effective_k = min(max_k, n_cand)

    # Similarity + topk
    similarity = q_emb @ c_emb.T  # [N_query, N_cand]
    _, topk_indices = torch.topk(similarity, k=effective_k, dim=1, sorted=True)
    del similarity

    # Build relevance matrix and n_relevant
    relevance = torch.zeros(n_query, effective_k, device=device)
    n_relevant_list = []
    all_rel_list = []
    max_n_rel = 0

    for i, rel_ids in enumerate(rel_cand_ids):
        rel_indices = {cand_id_to_idx[rid] for rid in rel_ids if rid in cand_id_to_idx}
        n_relevant_list.append(len(rel_indices))
        max_n_rel = max(max_n_rel, len(rel_indices))
        for j in range(effective_k):
            if topk_indices[i, j].item() in rel_indices:
                relevance[i, j] = 1.0
        # All rel scores (binary: all 1.0)
        all_rel_list.append([1.0] * len(rel_indices))

    n_relevant = torch.tensor(n_relevant_list, device=device, dtype=torch.float32)

    # Pad all_rel_scores to uniform length
    if max_n_rel == 0:
        max_n_rel = 1
    all_rel_scores = torch.zeros(n_query, max_n_rel, device=device)
    for i, scores in enumerate(all_rel_list):
        for j, s in enumerate(scores):
            all_rel_scores[i, j] = s

    # Pad relevance if effective_k < max_k
    if effective_k < max_k:
        pad = torch.zeros(n_query, max_k - effective_k, device=device)
        relevance = torch.cat([relevance, pad], dim=1)

    return metric_computer.compute(relevance, n_relevant, all_rel_scores)


def compute_classification_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    label_ids: list[str],
    candidate_ids: list[str],
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute classification metrics (top-k accuracy).

    This is a special case of retrieval where each query has exactly one label.

    Args:
        query_embeddings: Query embeddings [N_query, D]
        candidate_embeddings: Candidate embeddings [N_cand, D]
        label_ids: List of ground truth candidate IDs for each query
        candidate_ids: List of candidate IDs [N_cand]
        metrics: Complete metric names to compute

    Returns:
        Dictionary of metric scores (Hit@K = Accuracy@K)
    """
    rel_cand_ids = [[lid] for lid in label_ids]
    return compute_retrieval_metrics(
        query_embeddings,
        candidate_embeddings,
        rel_cand_ids,
        candidate_ids,
        metrics,
    )


# ============================================================================
# Public exports.
# ============================================================================

__all__ = [
    "DEFAULT_RETRIEVAL_METRICS",
    "RankingMetrics",
    "RankingMetricSpec",
    "compute_retrieval_metrics",
    "compute_classification_metrics",
    "parse_ranking_metric_names",
]
