"""Tests for RetrievalEvaluator helper methods and distributed protocol wiring."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_evaluator(metrics=("hit@1", "hit@3")):
    """Create a minimal evaluator for testing helper methods."""
    from vlm2emb.evaluation.evaluators.retrieval import RetrievalEvaluator
    from vlm2emb.evaluation.metrics import RankingMetrics

    evaluator = RetrievalEvaluator.__new__(RetrievalEvaluator)
    evaluator.metrics = tuple(metrics)
    evaluator._metrics_computer = RankingMetrics(metrics=evaluator.metrics)
    evaluator.dataset = MagicMock()
    evaluator.dataset.name = "ToyRetrieval"
    evaluator.batch_size = 2
    evaluator.num_workers = 0
    evaluator.show_progress = False
    return evaluator


def _make_runtime(*, process_index: int, num_processes: int, is_main_process: bool | None = None):
    class RuntimeFacade:
        def __init__(self):
            self.device = torch.device("cpu")
            self.process_index = process_index
            self.num_processes = num_processes
            self.is_main_process = process_index == 0 if is_main_process is None else is_main_process
            self.prepare_called = False
            self.gather_called = False

        def autocast(self):
            return nullcontext()

        def wait_for_everyone(self):
            return None

        def prepare(self, *args, **kwargs):
            self.prepare_called = True
            raise AssertionError("RetrievalEvaluator must not call accelerator.prepare()")

        def gather_for_metrics(self, *args, **kwargs):
            self.gather_called = True
            raise AssertionError("RetrievalEvaluator must not call accelerator.gather_for_metrics()")

    return RuntimeFacade()


class _ToyDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return {
            "text": f"sample-{index}",
            "images": [],
            "value": float(index),
        }


class _ToyModel(torch.nn.Module):
    def forward(self, values):
        values = values.float().view(-1, 1)
        return {"embeddings": torch.cat([values, values + 100.0], dim=1)}


class _ToyCollator:
    def __init__(self, processor_wrapper):
        self.processor_wrapper = processor_wrapper

    def __call__(self, batch):
        values = torch.tensor([item["value"] for item in batch], dtype=torch.float32)
        return {"inputs": {"values": values}}


def _simulate_block_cyclic_gather_payload(
    total_size: int,
    world_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    from vlm2emb.evaluation.distributed.retrieval_protocol import (
        build_shard_position_tensor,
        build_shard_spec,
    )

    shards = [
        build_shard_spec(
            total_size=total_size,
            world_size=world_size,
            rank=rank,
            block_size=block_size,
        )
        for rank in range(world_size)
    ]
    gather_count = max((shard.gather_count for shard in shards), default=0)
    gathered_embeds: list[torch.Tensor] = []
    gathered_positions: list[torch.Tensor] = []

    for shard in shards:
        local_embeds = torch.tensor(
            [[float(index), float(index + 100)] for index in shard.sample_indices],
            dtype=torch.float32,
        )
        if shard.real_count == 0:
            local_embeds = torch.empty((0, 2), dtype=torch.float32)
        local_positions = build_shard_position_tensor(shard)

        padded_embeds = torch.full((gather_count, 2), 999.0, dtype=torch.float32)
        padded_positions = torch.full((gather_count,), -1, dtype=torch.int64)
        if shard.real_count:
            padded_embeds[: shard.real_count] = local_embeds
            padded_positions[: shard.real_count] = local_positions
        gathered_embeds.append(padded_embeds)
        gathered_positions.append(padded_positions)

    return (
        torch.cat(gathered_embeds, dim=0),
        torch.cat(gathered_positions, dim=0),
        [shard.real_count for shard in shards],
    )


class TestEncodeWithDistributedProtocol:
    def test_single_process_passthrough(self, monkeypatch):
        from vlm2emb.evaluation.evaluators import retrieval as retrieval_mod

        evaluator = _make_evaluator()
        monkeypatch.setattr(retrieval_mod, "EvalCollator", _ToyCollator)

        result = evaluator._encode_with_dataloader(
            model=_ToyModel(),
            processor_wrapper=object(),
            accelerator=None,
            dataset=_ToyDataset(3),
            desc="queries",
            batch_size=2,
            num_workers=0,
        )

        assert result.shape == (3, 2)
        assert result.tolist() == [[0.0, 100.0], [1.0, 101.0], [2.0, 102.0]]

    def test_multi_process_main_rank_restores_original_order_after_block_cyclic_gather(self, monkeypatch):
        from vlm2emb.evaluation.distributed.retrieval_protocol import (
            build_shard_spec as real_build_shard_spec,
        )
        from vlm2emb.evaluation.evaluators import retrieval as retrieval_mod

        evaluator = _make_evaluator()
        runtime = _make_runtime(process_index=0, num_processes=2)
        calls: dict[str, list[object]] = {"shards": [], "gathers": []}
        gathered_embeds, gathered_positions, counts = _simulate_block_cyclic_gather_payload(
            total_size=10,
            world_size=2,
            block_size=2,
        )

        monkeypatch.setattr(retrieval_mod, "EvalCollator", _ToyCollator)

        def fake_build_shard_spec(total_size, world_size, rank, block_size=None):
            calls["shards"].append((total_size, world_size, rank, block_size))
            return real_build_shard_spec(total_size, world_size, rank, block_size=block_size)

        def fake_gather_tensor_first_dim(local_tensor):
            calls["gathers"].append((tuple(local_tensor.shape), str(local_tensor.dtype)))
            if local_tensor.dtype == torch.int64:
                return gathered_positions.clone(), counts
            return gathered_embeds.clone(), counts

        monkeypatch.setattr(retrieval_mod, "build_shard_spec", fake_build_shard_spec)
        monkeypatch.setattr(retrieval_mod, "gather_tensor_first_dim", fake_gather_tensor_first_dim)

        result = evaluator._encode_with_dataloader(
            model=_ToyModel(),
            processor_wrapper=object(),
            accelerator=cast(Any, runtime),
            dataset=_ToyDataset(10),
            desc="queries",
            batch_size=2,
            num_workers=0,
        )

        assert calls["shards"] == [(10, 2, 0, 2)]
        assert calls["gathers"] == [((6, 2), "torch.float32"), ((6,), "torch.int64")]
        assert result.shape == (10, 2)
        assert result.tolist() == [
            [0.0, 100.0],
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
        ]
        assert runtime.prepare_called is False
        assert runtime.gather_called is False

    def test_non_main_rank_returns_empty_tensor_for_empty_shard(self, monkeypatch):
        from vlm2emb.evaluation.evaluators import retrieval as retrieval_mod

        evaluator = _make_evaluator()
        runtime = _make_runtime(process_index=3, num_processes=4, is_main_process=False)
        gathered_embeds, gathered_positions, counts = _simulate_block_cyclic_gather_payload(
            total_size=5,
            world_size=4,
            block_size=2,
        )

        monkeypatch.setattr(retrieval_mod, "EvalCollator", _ToyCollator)

        def fake_gather_tensor_first_dim(local_tensor):
            if local_tensor.dtype == torch.int64:
                return gathered_positions.clone(), counts
            return gathered_embeds.clone(), counts

        monkeypatch.setattr(retrieval_mod, "gather_tensor_first_dim", fake_gather_tensor_first_dim)

        result = evaluator._encode_with_dataloader(
            model=_ToyModel(),
            processor_wrapper=object(),
            accelerator=cast(Any, runtime),
            dataset=_ToyDataset(5),
            desc="queries",
            batch_size=2,
            num_workers=0,
        )

        assert result.shape == (0, 2)
        assert runtime.prepare_called is False
        assert runtime.gather_called is False

    def test_requires_process_index_on_runtime_facade(self, monkeypatch):
        from vlm2emb.evaluation.evaluators import retrieval as retrieval_mod

        evaluator = _make_evaluator()
        runtime = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=2,
            is_main_process=True,
            autocast=lambda: nullcontext(),
            wait_for_everyone=lambda: None,
        )
        monkeypatch.setattr(retrieval_mod, "EvalCollator", _ToyCollator)

        with pytest.raises(TypeError, match="process_index"):
            evaluator._encode_with_dataloader(
                model=_ToyModel(),
                processor_wrapper=object(),
                accelerator=cast(Any, runtime),
                dataset=_ToyDataset(3),
                desc="queries",
                batch_size=2,
                num_workers=0,
            )

    def test_protocol_state_debug_logs_on_default_rank_only(self, monkeypatch):
        from vlm2emb.evaluation.distributed.retrieval_protocol import build_shard_spec
        from vlm2emb.evaluation.evaluators import retrieval as retrieval_mod

        evaluator = _make_evaluator()
        captured: list[dict[str, Any]] = []

        def fake_debug(msg, *args, **kwargs):
            captured.append(
                {
                    "message": msg % args if args else msg,
                    "kwargs": kwargs,
                }
            )

        monkeypatch.setattr(retrieval_mod.logger, "debug", fake_debug)

        evaluator._log_protocol_state(
            desc="queries",
            stage="post-trim",
            shard=build_shard_spec(total_size=5, world_size=2, rank=1),
            local_count=2,
            gather_counts=[3, 2],
            tensor_shape=(5, 2),
        )

        assert len(captured) == 1
        assert captured[0]["kwargs"] == {}
        assert "stage=post-trim" in captured[0]["message"]


class TestRetrievalEvaluatorCall:
    def test_broadcast_scores_uses_global_rank_zero(self, monkeypatch):
        from vlm2emb.evaluation.evaluators import retrieval as retrieval_mod

        evaluator = _make_evaluator(metrics=("hit@1",))
        runtime = _make_runtime(process_index=1, num_processes=2, is_main_process=False)
        captured: dict[str, Any] = {}

        def fake_broadcast_object_list(payload, from_process):
            captured["payload"] = payload
            captured["from_process"] = from_process
            return [{"hit@1": 0.875}]

        monkeypatch.setattr(
            retrieval_mod,
            "broadcast_object_list",
            fake_broadcast_object_list,
        )

        scores = evaluator._broadcast_scores({}, cast(Any, runtime))

        assert scores == {"hit@1": 0.875}
        assert captured == {"payload": [{}], "from_process": 0}

    def test_non_main_rank_returns_broadcast_scores(self, monkeypatch):
        evaluator = _make_evaluator(metrics=("hit@1",))
        evaluator.dataset.queries = _ToyDataset(1)
        evaluator.dataset.candidates = _ToyDataset(1)
        runtime = _make_runtime(process_index=1, num_processes=2, is_main_process=False)

        def fake_encode(*args, **kwargs):
            return torch.empty((0, 2), dtype=torch.float32)

        def fail_compute(*args, **kwargs):
            raise AssertionError("non-main rank must not compute retrieval metrics")

        def fake_broadcast(scores, accelerator):
            assert scores == {}
            assert accelerator is runtime
            return {"hit@1": 0.875}

        monkeypatch.setattr(evaluator, "_encode_with_dataloader", fake_encode)
        monkeypatch.setattr(evaluator, "_compute_predictions", fail_compute)
        monkeypatch.setattr(evaluator, "_broadcast_scores", fake_broadcast)

        scores = evaluator(
            model=_ToyModel(),
            processor_wrapper=object(),
            accelerator=cast(Any, runtime),
        )

        assert scores == {"hit@1": 0.875}

    def test_main_rank_broadcasts_computed_scores(self, monkeypatch):
        evaluator = _make_evaluator(metrics=("hit@1",))
        evaluator.dataset.queries = _ToyDataset(1)
        evaluator.dataset.candidates = _ToyDataset(1)
        runtime = _make_runtime(process_index=0, num_processes=2, is_main_process=True)
        computed_scores = {"hit@1": 0.625}

        def fake_encode(*args, **kwargs):
            return torch.ones((1, 2), dtype=torch.float32)

        def fake_predictions(query_embeds, cand_embeds):
            return {
                "relevance": torch.ones((1, 1)),
                "n_relevant": torch.ones((1,)),
                "all_rel_scores": torch.ones((1, 1)),
            }

        def fake_broadcast(scores, accelerator):
            assert scores == computed_scores
            assert accelerator is runtime
            return scores

        evaluator._metrics_computer = MagicMock()
        evaluator._metrics_computer.compute = MagicMock(return_value=computed_scores)
        monkeypatch.setattr(evaluator, "_encode_with_dataloader", fake_encode)
        monkeypatch.setattr(evaluator, "_compute_predictions", fake_predictions)
        monkeypatch.setattr(evaluator, "_broadcast_scores", fake_broadcast)

        scores = evaluator(
            model=_ToyModel(),
            processor_wrapper=object(),
            accelerator=cast(Any, runtime),
        )

        assert scores == computed_scores


class TestOrderedTrimmedEmbeddings:
    """Test the post-gather ordered embedding contract."""

    def test_trimmed_gather_matches_total_size(self, device):
        from vlm2emb.evaluation.distributed.retrieval_protocol import trim_gathered_tensor

        gathered = torch.tensor(
            [
                [0.0, 100.0],
                [1.0, 101.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
                [999.0, 999.0],
            ],
            device=device,
        )

        result = trim_gathered_tensor(gathered, counts=[3, 2], total_size=5)

        assert result.device == gathered.device
        assert result.shape == (5, 2)
        assert result.tolist() == [
            [0.0, 100.0],
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
        ]


class TestComputePredictions:
    """Test _compute_predictions with mock Lance dataset."""

    def _setup_mock_dataset(self, evaluator, cand_ids, query_ids, qrel_data, device):
        import pyarrow as pa

        cand_table = pa.table({"id": cand_ids})
        evaluator.dataset.candidates.to_table = MagicMock(return_value=cand_table)

        query_table = pa.table({"id": query_ids})
        evaluator.dataset.queries.to_table = MagicMock(return_value=query_table)

        qrel_query_ids = [d[0] for d in qrel_data]
        qrel_modes = [d[1] for d in qrel_data]
        qrel_cand_ids_col = [d[2] for d in qrel_data]
        qrel_cand_scores_col = [d[3] for d in qrel_data]

        qrels_table = pa.table(
            {
                "query_id": qrel_query_ids,
                "mode": qrel_modes,
                "candidate_ids": qrel_cand_ids_col,
                "candidate_scores": qrel_cand_scores_col,
            }
        )
        evaluator.dataset.qrels.to_table = MagicMock(return_value=qrels_table)

    def test_sparse_basic(self, device):
        evaluator = _make_evaluator(metrics=("hit@1", "hit@3"))

        cand_ids = ["c0", "c1", "c2", "c3"]
        query_ids = ["q0", "q1"]
        qrel_data = [
            ("q0", "sparse", ["c0", "c2"], [1.0, 1.0]),
            ("q1", "sparse", ["c1"], [1.0]),
        ]
        self._setup_mock_dataset(evaluator, cand_ids, query_ids, qrel_data, device)

        q_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        c_emb = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [-1.0, -1.0],
            ],
            device=device,
        )

        result = evaluator._compute_predictions(q_emb, c_emb)

        assert "relevance" in result
        assert "n_relevant" in result
        assert "all_rel_scores" in result
        assert result["relevance"].shape == (2, 3)
        assert result["n_relevant"].shape == (2,)
        assert result["n_relevant"][0].item() == 2
        assert result["n_relevant"][1].item() == 1

    def test_k_larger_than_candidates(self, device):
        evaluator = _make_evaluator(metrics=("hit@1", "hit@10"))

        cand_ids = ["c0", "c1", "c2"]
        query_ids = ["q0"]
        qrel_data = [("q0", "sparse", ["c0"], [1.0])]
        self._setup_mock_dataset(evaluator, cand_ids, query_ids, qrel_data, device)

        q_emb = torch.tensor([[1.0, 0.0]], device=device)
        c_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], device=device)

        result = evaluator._compute_predictions(q_emb, c_emb)
        assert result["relevance"].shape == (1, 10)
        assert result["relevance"][0, 3:].sum().item() == 0.0

    def test_qrel_query_mismatch_raises(self, device):
        evaluator = _make_evaluator()

        cand_ids = ["c0"]
        query_ids = ["q0", "q1"]
        qrel_data = [("q0", "sparse", ["c0"], [1.0])]
        self._setup_mock_dataset(evaluator, cand_ids, query_ids, qrel_data, device)

        q_emb = torch.tensor([[1.0], [0.0]], device=device)
        c_emb = torch.tensor([[1.0]], device=device)

        with pytest.raises(ValueError, match="qrels count"):
            evaluator._compute_predictions(q_emb, c_emb)

    def test_exhaustive_mode(self, device):
        evaluator = _make_evaluator(metrics=("hit@1", "hit@2"))

        cand_ids = ["c0", "c1", "c2", "c3"]
        query_ids = ["q0"]
        qrel_data = [("q0", "exhaustive", ["c0", "c1", "c2"], [1.0, 0.0, 1.0])]
        self._setup_mock_dataset(evaluator, cand_ids, query_ids, qrel_data, device)

        q_emb = torch.tensor([[0.5, 0.5]], device=device)
        c_emb = torch.tensor(
            [
                [0.4, 0.4],
                [0.0, 0.0],
                [0.3, 0.3],
                [1.0, 1.0],
            ],
            device=device,
        )

        result = evaluator._compute_predictions(q_emb, c_emb)
        assert result["relevance"].shape == (1, 2)
        assert result["n_relevant"][0].item() == 2

    def test_missing_candidate_id_in_qrels_raises(self, device):
        evaluator = _make_evaluator()

        cand_ids = ["c0"]
        query_ids = ["q0"]
        qrel_data = [("q0", "sparse", ["missing-candidate"], [1.0])]
        self._setup_mock_dataset(evaluator, cand_ids, query_ids, qrel_data, device)

        q_emb = torch.tensor([[1.0]], device=device)
        c_emb = torch.tensor([[1.0]], device=device)

        with pytest.raises(ValueError, match="missing candidate ids"):
            evaluator._compute_predictions(q_emb, c_emb)
