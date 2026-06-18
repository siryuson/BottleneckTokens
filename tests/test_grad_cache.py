from __future__ import annotations

from typing import Any

import pytest
import torch

from vlm2emb.training.grad_cache import ChunkContext, grad_cache_accumulate


def _sample(sample_id: str, length: int) -> dict[str, Any]:
    return {
        "id": sample_id,
        "length": length,
        "input_ids": torch.ones((1, length), dtype=torch.long),
    }


def _encode(chunk: list[dict[str, Any]]) -> torch.Tensor:
    values = [float(sample["value"]) for sample in chunk]
    return torch.tensor(values, dtype=torch.float32).view(-1, 1)


def test_grad_cache_aligned_sorting_is_stable_ascending_and_non_mutating() -> None:
    queries = [
        {"id": "q0", "length": 5, "value": 0.0},
        {"id": "q1", "length": 2, "value": 1.0},
        {"id": "q2", "length": 7, "value": 2.0},
        {"id": "q3", "length": 2, "value": 3.0},
    ]
    positives = [
        {"id": "p0", "length": 50, "value": 10.0},
        {"id": "p1", "length": 20, "value": 11.0},
        {"id": "p2", "length": 70, "value": 12.0},
        {"id": "p3", "length": 10, "value": 13.0},
    ]
    original_query_order = [sample["id"] for sample in queries]
    original_positive_order = [sample["id"] for sample in positives]
    seen_chunks: list[tuple[int, list[str], list[float]]] = []

    def forward_backward(
        chunk: list[dict[str, Any]],
        cache: torch.Tensor,
        ctx: ChunkContext,
    ) -> None:
        seen_chunks.append(
            (ctx.group_idx, [sample["id"] for sample in chunk], cache.flatten().tolist())
        )

    loss, _ = grad_cache_accumulate(
        inputs=[queries, positives],
        encode_fns=[_encode, _encode],
        loss_fn=lambda *reps: sum(rep.sum() for rep in reps),
        chunk_sizes=[2, 2],
        forward_backward_fns=[forward_backward, forward_backward],
        aligned_sort_key_fn=lambda sample: sample["length"],
        aligned_sort_group_idx=0,
    )

    assert loss.item() == pytest.approx(52.0)
    assert [sample["id"] for sample in queries] == original_query_order
    assert [sample["id"] for sample in positives] == original_positive_order
    assert seen_chunks == [
        (0, ["q1", "q3"], [1.0, 1.0]),
        (0, ["q0", "q2"], [1.0, 1.0]),
        (1, ["p1", "p3"], [1.0, 1.0]),
        (1, ["p0", "p2"], [1.0, 1.0]),
    ]


def test_grad_cache_without_sort_key_preserves_input_order() -> None:
    queries = [
        {"id": "q0", "length": 5, "value": 0.0},
        {"id": "q1", "length": 2, "value": 1.0},
        {"id": "q2", "length": 7, "value": 2.0},
    ]
    positives = [
        {"id": "p0", "length": 50, "value": 10.0},
        {"id": "p1", "length": 20, "value": 11.0},
        {"id": "p2", "length": 70, "value": 12.0},
    ]
    seen_chunks: list[list[str]] = []

    def forward_backward(
        chunk: list[dict[str, Any]],
        cache: torch.Tensor,
        ctx: ChunkContext,
    ) -> None:
        seen_chunks.append([sample["id"] for sample in chunk])

    grad_cache_accumulate(
        inputs=[queries, positives],
        encode_fns=[_encode, _encode],
        loss_fn=lambda *reps: sum(rep.sum() for rep in reps),
        chunk_sizes=[2, 2],
        forward_backward_fns=[forward_backward, forward_backward],
    )

    assert seen_chunks == [["q0", "q1"], ["q2"], ["p0", "p1"], ["p2"]]


def test_grad_cache_aligned_sorting_rejects_mismatched_group_lengths() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        grad_cache_accumulate(
            inputs=[
                [{"id": "q0", "length": 1, "value": 0.0}],
                [
                    {"id": "p0", "length": 1, "value": 10.0},
                    {"id": "p1", "length": 1, "value": 11.0},
                ],
            ],
            encode_fns=[_encode, _encode],
            loss_fn=lambda *reps: sum(rep.sum() for rep in reps),
            chunk_sizes=[1, 1],
            aligned_sort_key_fn=lambda sample: sample["length"],
        )


def test_grad_cache_aligned_sorting_allows_empty_groups() -> None:
    loss, group_results = grad_cache_accumulate(
        inputs=[[], []],
        encode_fns=[_encode, _encode],
        loss_fn=lambda *reps: sum(rep.sum() for rep in reps),
        chunk_sizes=[1, 1],
        aligned_sort_key_fn=lambda sample: sample["length"],
    )

    assert loss.item() == 0.0
    assert group_results == [[], []]


def test_grad_cache_chunk_context_uses_active_group_ordinals() -> None:
    contexts: list[ChunkContext] = []

    def forward_backward(
        chunk: list[dict[str, Any]],
        cache: torch.Tensor,
        ctx: ChunkContext,
    ) -> None:
        contexts.append(ctx)

    grad_cache_accumulate(
        inputs=[
            [{"id": "g0", "value": 0.0}],
            [],
            [{"id": "g2a", "value": 2.0}, {"id": "g2b", "value": 3.0}],
        ],
        encode_fns=[_encode, _encode, _encode],
        loss_fn=lambda *reps: sum(rep.sum() for rep in reps),
        chunk_sizes=[1, 1, 1],
        forward_backward_fns=[forward_backward, forward_backward, forward_backward],
    )

    assert contexts == [
        ChunkContext(group_idx=0, chunk_idx=0, num_groups=2, num_chunks=1),
        ChunkContext(group_idx=1, chunk_idx=0, num_groups=2, num_chunks=2),
        ChunkContext(group_idx=1, chunk_idx=1, num_groups=2, num_chunks=2),
    ]
