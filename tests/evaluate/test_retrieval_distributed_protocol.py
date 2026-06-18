from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import vlm2emb.evaluation.distributed.retrieval_protocol as retrieval_protocol
from vlm2emb.evaluation.distributed import (
    build_shard_position_tensor,
    build_shard_spec,
    build_sharded_dataset_view,
    gather_tensor_first_dim,
    reorder_gathered_tensor,
    trim_gathered_tensor,
)


def _simulate_block_cyclic_gather(
    total_size: int,
    world_size: int,
    block_size: int,
    *,
    feature_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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
            [[float(index * feature_dim + offset) for offset in range(feature_dim)] for index in shard.sample_indices],
            dtype=torch.float32,
        )
        if shard.real_count == 0:
            local_embeds = torch.empty((0, feature_dim), dtype=torch.float32)
        local_positions = build_shard_position_tensor(shard)

        padded_embeds = torch.full((gather_count, feature_dim), -1.0, dtype=torch.float32)
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


def test_build_shard_spec_keeps_contiguous_default_for_existing_callers():
    shard0 = build_shard_spec(total_size=5, world_size=2, rank=0)
    shard1 = build_shard_spec(total_size=5, world_size=2, rank=1)

    assert shard0.block_size is None
    assert shard0.block_ranges == ((0, 3),)
    assert shard0.sample_indices == (0, 1, 2)
    assert shard0.gather_count == 3
    assert shard0.start == 0
    assert shard0.stop == 3

    assert shard1.block_ranges == ((3, 5),)
    assert shard1.sample_indices == (3, 4)
    assert shard1.gather_count == 3
    assert shard1.start == 3
    assert shard1.stop == 5


def test_build_shard_spec_block_cyclic_assigns_batch_sized_blocks():
    shards = [
        build_shard_spec(total_size=10, world_size=3, rank=rank, block_size=2)
        for rank in range(3)
    ]

    assert [shard.block_ranges for shard in shards] == [
        ((0, 2), (6, 8)),
        ((2, 4), (8, 10)),
        ((4, 6),),
    ]
    assert [shard.sample_indices for shard in shards] == [
        (0, 1, 6, 7),
        (2, 3, 8, 9),
        (4, 5),
    ]
    assert [shard.real_count for shard in shards] == [4, 4, 2]
    assert all(shard.gather_count == 4 for shard in shards)


def test_build_shard_spec_block_cyclic_handles_tail_block_and_empty_ranks():
    shards = [
        build_shard_spec(total_size=5, world_size=4, rank=rank, block_size=2)
        for rank in range(4)
    ]

    assert [shard.block_ranges for shard in shards] == [
        ((0, 2),),
        ((2, 4),),
        ((4, 5),),
        (),
    ]
    assert [shard.sample_indices for shard in shards] == [
        (0, 1),
        (2, 3),
        (4,),
        (),
    ]
    assert [shard.real_count for shard in shards] == [2, 2, 1, 0]
    assert all(shard.gather_count == 2 for shard in shards)


def test_build_sharded_dataset_view_uses_explicit_index_list():
    dataset = list(range(10))
    shard = build_shard_spec(total_size=10, world_size=3, rank=0, block_size=2)

    subset = build_sharded_dataset_view(dataset, shard)

    assert len(subset) == 4
    assert [subset[index] for index in range(len(subset))] == [0, 1, 6, 7]


def test_build_shard_position_tensor_matches_sample_indices():
    shard = build_shard_spec(total_size=10, world_size=3, rank=1, block_size=2)

    positions = build_shard_position_tensor(shard)

    assert positions.dtype == torch.int64
    assert positions.tolist() == [2, 3, 8, 9]


def test_gather_tensor_first_dim_single_process_passthrough():
    local = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    gathered, counts = gather_tensor_first_dim(local)

    assert counts == [3]
    assert torch.equal(gathered, local)


def test_resolve_trailing_shape_uses_non_empty_rank_shapes(monkeypatch):
    local = torch.empty((0, 0), dtype=torch.float32)
    all_gather_calls = 0

    def fake_all_gather(outputs, input_tensor):
        nonlocal all_gather_calls
        all_gather_calls += 1
        values = [1, 1, 1] if all_gather_calls == 1 else [1536, 0, 1536]
        for output, value in zip(outputs, values, strict=True):
            output.copy_(torch.tensor([value], dtype=input_tensor.dtype))

    monkeypatch.setattr(retrieval_protocol, "_is_distributed_initialized", lambda: True)
    monkeypatch.setattr(retrieval_protocol.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(retrieval_protocol.dist, "all_gather", fake_all_gather)

    trailing_shape = retrieval_protocol._resolve_trailing_shape(local, counts=[4, 0, 4])

    assert trailing_shape == (1536,)


def test_resolve_trailing_shape_handles_empty_rank_without_trailing_dims(monkeypatch):
    local = torch.empty((0,), dtype=torch.float32)
    all_gather_calls = 0

    def fake_all_gather(outputs, input_tensor):
        nonlocal all_gather_calls
        all_gather_calls += 1
        values = [0, 1, 1] if all_gather_calls == 1 else [-1, 1536, 1536]
        for output, value in zip(outputs, values, strict=True):
            output.copy_(torch.tensor([value], dtype=input_tensor.dtype))

    monkeypatch.setattr(retrieval_protocol, "_is_distributed_initialized", lambda: True)
    monkeypatch.setattr(retrieval_protocol.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(retrieval_protocol.dist, "all_gather", fake_all_gather)

    trailing_shape = retrieval_protocol._resolve_trailing_shape(local, counts=[0, 4, 4])

    assert trailing_shape == (1536,)


def test_resolve_trailing_shape_rejects_non_empty_rank_mismatch(monkeypatch):
    local = torch.empty((2, 1536), dtype=torch.float32)
    all_gather_calls = 0

    def fake_all_gather(outputs, input_tensor):
        nonlocal all_gather_calls
        all_gather_calls += 1
        values = [1, 1] if all_gather_calls == 1 else [1536, 1024]
        for output, value in zip(outputs, values, strict=True):
            output.copy_(torch.tensor([value], dtype=input_tensor.dtype))

    monkeypatch.setattr(retrieval_protocol, "_is_distributed_initialized", lambda: True)
    monkeypatch.setattr(retrieval_protocol.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(retrieval_protocol.dist, "all_gather", fake_all_gather)

    with pytest.raises(ValueError, match="incompatible trailing shapes"):
        retrieval_protocol._resolve_trailing_shape(local, counts=[2, 2])


def test_trim_gathered_tensor_uses_rank_major_counts_order():
    gathered = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [9.0, 9.0],
            [9.0, 9.0],
        ]
    )

    trimmed = trim_gathered_tensor(gathered, counts=[2, 1], total_size=3)

    assert trimmed.shape == (3, 2)
    assert trimmed.tolist() == [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]


def test_reorder_gathered_tensor_restores_original_order_after_block_cyclic_gather():
    gathered, positions, counts = _simulate_block_cyclic_gather(total_size=10, world_size=3, block_size=2)

    reordered = reorder_gathered_tensor(
        gathered,
        positions=positions,
        counts=counts,
        total_size=10,
    )

    expected = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    assert reordered.shape == (10, 2)
    assert torch.equal(reordered, expected)


def test_reorder_gathered_tensor_handles_tail_block_and_empty_rank_padding():
    gathered, positions, counts = _simulate_block_cyclic_gather(total_size=5, world_size=4, block_size=2)

    reordered = reorder_gathered_tensor(
        gathered,
        positions=positions,
        counts=counts,
        total_size=5,
    )

    assert counts == [2, 2, 1, 0]
    assert reordered.tolist() == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
        [8.0, 9.0],
    ]


def test_reorder_gathered_tensor_handles_total_size_smaller_than_world_size():
    gathered, positions, counts = _simulate_block_cyclic_gather(total_size=2, world_size=4, block_size=2)

    reordered = reorder_gathered_tensor(
        gathered,
        positions=positions,
        counts=counts,
        total_size=2,
    )

    assert counts == [2, 0, 0, 0]
    assert reordered.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_reorder_gathered_tensor_rejects_duplicate_positions():
    gathered = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ]
    )
    positions = torch.tensor([0, 1, 1, 2], dtype=torch.int64)

    with pytest.raises(ValueError, match="unique"):
        reorder_gathered_tensor(gathered, positions=positions, counts=[2, 2], total_size=4)


def test_package_exports_explicit_reorder_helpers():
    from vlm2emb import evaluation as evaluation_pkg

    assert hasattr(evaluation_pkg.distributed, "build_shard_position_tensor")
    assert hasattr(evaluation_pkg.distributed, "reorder_gathered_tensor")


def test_cpu_two_rank_gather_worker():
    if os.environ.get("VLM2EMB_CPU_GATHER_WORKER") != "1":
        pytest.skip("helper test only runs inside the optional torch.distributed subprocess")

    if dist.is_initialized():
        with suppress(Exception):
            dist.destroy_process_group()

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        shard = build_shard_spec(total_size=3, world_size=dist.get_world_size(), rank=rank, block_size=2)
        local = torch.tensor(
            [[float(index * 2), float(index * 2 + 1)] for index in shard.sample_indices],
            dtype=torch.float32,
        )
        if shard.real_count == 0:
            local = torch.empty((0, 2), dtype=torch.float32)
        local_positions = build_shard_position_tensor(shard)

        gathered, counts = gather_tensor_first_dim(local)
        gathered_positions, position_counts = gather_tensor_first_dim(local_positions)
        reordered = reorder_gathered_tensor(gathered, positions=gathered_positions, counts=counts, total_size=3)

        assert counts == [2, 1]
        assert position_counts == counts
        assert reordered.tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_cpu_two_rank_gather_smoke():
    if os.environ.get("VLM2EMB_RUN_CPU_DISTRIBUTED_SMOKE") != "1":
        pytest.skip("local validation keeps the CPU torchrun smoke as a defined but non-executed test")

    file_path = Path(__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "pytest",
        "-q",
        str(file_path),
        "-k",
        "test_cpu_two_rank_gather_worker",
        "-x",
    ]
    env = os.environ.copy()
    env["VLM2EMB_CPU_GATHER_WORKER"] = "1"

    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_nccl_smoke_explicit_protocol():
    if os.environ.get("VLM2EMB_RUN_NCCL_SMOKE") != "1":
        pytest.skip("NCCL smoke is opt-in and only runs on a real 2xGPU host")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the NCCL smoke")

    if dist.is_initialized():
        with suppress(Exception):
            dist.destroy_process_group()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        shard = build_shard_spec(total_size=3, world_size=world_size, rank=rank, block_size=2)
        local = torch.tensor(
            [[float(index * 2), float(index * 2 + 1)] for index in shard.sample_indices],
            device=torch.device("cuda", local_rank),
            dtype=torch.float32,
        )
        if shard.real_count == 0:
            local = torch.empty((0, 2), device=torch.device("cuda", local_rank), dtype=torch.float32)
        local_positions = build_shard_position_tensor(shard, device=torch.device("cuda", local_rank))

        gathered, counts = gather_tensor_first_dim(local)
        gathered_positions, position_counts = gather_tensor_first_dim(local_positions)
        reordered = reorder_gathered_tensor(gathered, positions=gathered_positions, counts=counts, total_size=3)

        print(
            "nccl_smoke "
            f"rank={rank} world_size={world_size} block_ranges={shard.block_ranges} "
            f"local_count={shard.real_count} gather_counts={counts} position_counts={position_counts} "
            f"trimmed_shape={tuple(reordered.shape)}"
        )
        assert reordered.shape == (3, 2)
    finally:
        dist.barrier()
        dist.destroy_process_group()
