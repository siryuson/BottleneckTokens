"""Tests for RankingMetrics GPU batch computation."""

from __future__ import annotations

import pytest
import torch

from vlm2emb.evaluation.metrics import (
    DEFAULT_RETRIEVAL_METRICS,
    RankingMetrics,
    parse_ranking_metric_names,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestHitAtK:
    def test_perfect_hit(self, device):
        # Query with relevant doc at position 0
        relevance = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.tensor([[1.0]], device=device)
        m = RankingMetrics(metrics=["hit@1", "hit@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["hit@1"] == 1.0
        assert scores["hit@3"] == 1.0

    def test_miss_at_1_hit_at_3(self, device):
        # Relevant doc at position 2 (0-indexed)
        relevance = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.tensor([[1.0]], device=device)
        m = RankingMetrics(metrics=["hit@1", "hit@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["hit@1"] == 0.0
        assert scores["hit@3"] == 1.0


class TestMRRAtK:
    """Tests for MRR with K truncation (bug fix: mrr@1 != mrr@10)."""

    def test_k_truncation(self, device):
        # First relevant at position 2 (0-indexed)
        relevance = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.tensor([[1.0]], device=device)
        m = RankingMetrics(metrics=["mrr@1", "mrr@10"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["mrr@1"] == 0.0  # No hit in top 1
        assert scores["mrr@10"] == pytest.approx(1.0 / 3)  # Hit at position 3

    def test_first_position_hit(self, device):
        relevance = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.tensor([[1.0]], device=device)
        m = RankingMetrics(metrics=["mrr@1", "mrr@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["mrr@1"] == 1.0
        assert scores["mrr@3"] == 1.0

    def test_no_hit(self, device):
        relevance = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.tensor([[1.0]], device=device)
        m = RankingMetrics(metrics=["mrr@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["mrr@3"] == 0.0


class TestMAPAtK:
    """Tests for MAP with K truncation (bug fix: map@1 != map@10)."""

    def test_k_truncation(self, device):
        # Relevant at positions 0 and 2 (0-indexed)
        relevance = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([2], device=device)
        all_rel = torch.ones(1, 2, device=device)
        m = RankingMetrics(metrics=["map@1", "map@10"])
        scores = m.compute(relevance, n_relevant, all_rel)
        # map@1: precision_at_1 * rel_1 / min(n_rel, 1) = (1/1)*1/1 = 1.0
        assert scores["map@1"] == pytest.approx(1.0)
        # map@10: (1/1*1 + 2/3*1) / min(2, 10) = (1 + 0.6667) / 2 = 0.8333
        assert scores["map@10"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


class TestNDCGAtK:
    def test_perfect_ranking(self, device):
        # All relevant docs at top
        relevance = torch.tensor([[1.0, 1.0, 0.0]], device=device)
        n_relevant = torch.tensor([2], device=device)
        all_rel = torch.ones(1, 2, device=device)
        m = RankingMetrics(metrics=["ndcg@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["ndcg@3"] == pytest.approx(1.0)

    def test_graded_relevance(self, device):
        # Graded relevance: actual order [3.0, 1.0], ideal [3.0, 2.0]
        relevance = torch.tensor([[3.0, 1.0, 0.0]], device=device)
        n_relevant = torch.tensor([2], device=device)
        all_rel = torch.tensor([[3.0, 2.0, 0.0]], device=device)
        m = RankingMetrics(metrics=["ndcg@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        # DCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309 = 3.6309
        # IDCG = 3/log2(2) + 2/log2(3) = 3 + 1.2619 = 4.2619
        expected = (3.0 + 1.0 / 1.585) / (3.0 + 2.0 / 1.585)
        assert scores["ndcg@3"] == pytest.approx(expected, rel=1e-3)

    def test_graded_not_used_for_binary_metrics(self, device):
        """Non-NDCG metrics should use binary_rel = (relevance > 0)."""
        relevance = torch.tensor([[3.0, 0.0, 2.0]], device=device)
        n_relevant = torch.tensor([2], device=device)
        all_rel = torch.tensor([[3.0, 2.0, 0.0]], device=device)
        m = RankingMetrics(metrics=["hit@3", "mrr@3", "map@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        # Binary: positions 0 and 2 are relevant (>0)
        assert scores["hit@3"] == 1.0
        assert scores["mrr@3"] == pytest.approx(1.0)
        assert scores["map@3"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


class TestEmptyGroundTruth:
    def test_empty_gt_returns_zero(self, device):
        relevance = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([0], device=device)
        all_rel = torch.tensor([[0.0]], device=device)
        m = RankingMetrics(metrics=["hit@1", "hit@3"])
        scores = m.compute(relevance, n_relevant, all_rel)
        for key, val in scores.items():
            assert val == pytest.approx(0.0, abs=1e-6), f"{key} should be 0.0 for empty GT"

    def test_empty_gt_warning(self, device, caplog):
        import logging
        relevance = torch.tensor([[0.0, 0.0]], device=device)
        n_relevant = torch.tensor([0], device=device)
        all_rel = torch.tensor([[0.0]], device=device)
        m = RankingMetrics(metrics=["hit@1"])
        with caplog.at_level(logging.WARNING):
            m.compute(relevance, n_relevant, all_rel)
        assert "1 queries have empty ground truth" in caplog.text


class TestCompleteMetricNames:
    def test_only_requested_metrics_are_returned(self, device):
        relevance = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]], device=device)
        n_relevant = torch.tensor([3], device=device)
        all_rel = torch.ones(1, 3, device=device)
        m = RankingMetrics(metrics=["hit@1", "ndcg@5"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert set(scores.keys()) == {"hit@1", "ndcg@5"}
        assert m.max_k == 5
        assert m.metrics == ("hit@1", "ndcg@5")

    def test_default_metrics_match_explicit_default(self, device):
        relevance = torch.tensor([[1.0] + [0.0] * 9], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.ones(1, 1, device=device)
        m = RankingMetrics()
        scores = m.compute(relevance, n_relevant, all_rel)
        assert set(scores.keys()) == set(DEFAULT_RETRIEVAL_METRICS)
        assert m.max_k == 10

    def test_duplicate_metric_names_are_deduplicated(self):
        specs = parse_ranking_metric_names(["hit@1", "hit@1", "ndcg@5"])
        assert tuple(spec.name for spec in specs) == ("hit@1", "ndcg@5")

    def test_relevance_width_smaller_than_requested_k_raises(self, device):
        relevance = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        n_relevant = torch.tensor([1], device=device)
        all_rel = torch.tensor([[1.0]], device=device)
        m = RankingMetrics(metrics=["hit@5"])

        with pytest.raises(ValueError, match="relevance width"):
            m.compute(relevance, n_relevant, all_rel)

    @pytest.mark.parametrize(
        "metric_name",
        ["hit", "hit@0", "ndcg@abc", "foo@1", "precision@1", "recall@1"],
    )
    def test_invalid_metric_name_raises(self, metric_name):
        with pytest.raises(ValueError, match="metric|Metric|k|Unknown|expected"):
            parse_ranking_metric_names([metric_name])


class TestMultipleQueries:
    def test_mean_across_queries(self, device):
        # Query 0: hit at position 0, Query 1: no hit
        relevance = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], device=device)
        n_relevant = torch.tensor([1, 1], device=device)
        all_rel = torch.ones(2, 1, device=device)
        m = RankingMetrics(metrics=["hit@1"])
        scores = m.compute(relevance, n_relevant, all_rel)
        assert scores["hit@1"] == pytest.approx(0.5)  # Mean of [1.0, 0.0]


class TestComputeRetrievalMetricsCompat:
    """Tests for the numpy-based compatibility adapter."""

    def test_basic_retrieval(self):
        import numpy as np

        from vlm2emb.evaluation.metrics import compute_retrieval_metrics

        # 3 queries, 5 candidates, identity-like embeddings
        np.random.seed(42)
        n_query, n_cand, dim = 3, 5, 8
        q = np.random.randn(n_query, dim).astype(np.float32)
        c = np.random.randn(n_cand, dim).astype(np.float32)
        cand_ids = [f"c{i}" for i in range(n_cand)]
        # Each query has 1 relevant candidate
        rel_cand_ids = [["c0"], ["c1"], ["c2"]]

        scores = compute_retrieval_metrics(
            q,
            c,
            rel_cand_ids,
            cand_ids,
            metrics=["hit@1", "ndcg@3"],
        )
        # Should return dict[str, float] with expected keys
        assert isinstance(scores, dict)
        assert "hit@1" in scores
        assert "ndcg@3" in scores
        assert isinstance(scores["hit@1"], float)

    def test_k_larger_than_candidates(self):
        import numpy as np

        from vlm2emb.evaluation.metrics import compute_retrieval_metrics

        # 2 queries, 2 candidates, dim=4
        q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        cand_ids = ["c0", "c1"]
        rel_cand_ids = [["c0"], ["c1"]]

        # k=5 > n_cand=2, should not crash
        scores = compute_retrieval_metrics(
            q,
            c,
            rel_cand_ids,
            cand_ids,
            metrics=["hit@1", "hit@5"],
        )
        assert "hit@5" in scores
        assert isinstance(scores["hit@5"], float)

    def test_no_relevant_candidates(self):
        import numpy as np

        from vlm2emb.evaluation.metrics import compute_retrieval_metrics

        q = np.ones((1, 4), dtype=np.float32)
        c = np.ones((3, 4), dtype=np.float32)
        cand_ids = ["c0", "c1", "c2"]
        rel_cand_ids = [["nonexistent"]]  # no match in cand_ids

        scores = compute_retrieval_metrics(q, c, rel_cand_ids, cand_ids, metrics=["hit@1"])
        assert scores["hit@1"] == pytest.approx(0.0)
