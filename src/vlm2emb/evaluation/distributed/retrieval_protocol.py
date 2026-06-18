"""Explicit shard and gather helpers for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Subset


@dataclass(frozen=True)
class ShardSpec:
    total_size: int
    world_size: int
    rank: int
    block_size: int | None
    block_ranges: tuple[tuple[int, int], ...]
    sample_indices: tuple[int, ...]
    real_count: int
    gather_count: int

    @property
    def start(self) -> int:
        """Return the minimum covered index for compatibility with older callers."""
        if not self.block_ranges:
            return self.total_size
        return self.block_ranges[0][0]

    @property
    def stop(self) -> int:
        """Return the maximum exclusive covered index for compatibility with older callers."""
        if not self.block_ranges:
            return self.total_size
        return self.block_ranges[-1][1]


def _build_contiguous_shard_spec(total_size: int, world_size: int, rank: int) -> ShardSpec:
    base = total_size // world_size
    remainder = total_size % world_size
    start = rank * base + min(rank, remainder)
    real_count = base + (1 if rank < remainder else 0)
    stop = start + real_count
    gather_count = ceil(total_size / world_size) if total_size > 0 else 0
    sample_indices = tuple(range(start, stop))
    block_ranges = ((start, stop),) if real_count > 0 else ()
    return ShardSpec(
        total_size=total_size,
        world_size=world_size,
        rank=rank,
        block_size=None,
        block_ranges=block_ranges,
        sample_indices=sample_indices,
        real_count=real_count,
        gather_count=gather_count,
    )


def _build_block_cyclic_shard_spec(
    total_size: int,
    world_size: int,
    rank: int,
    block_size: int,
) -> ShardSpec:
    counts = [0] * world_size
    local_block_ranges: list[tuple[int, int]] = []
    local_sample_indices: list[int] = []

    for block_idx, start in enumerate(range(0, total_size, block_size)):
        stop = min(start + block_size, total_size)
        owner_rank = block_idx % world_size
        counts[owner_rank] += stop - start
        if owner_rank == rank:
            local_block_ranges.append((start, stop))
            local_sample_indices.extend(range(start, stop))

    return ShardSpec(
        total_size=total_size,
        world_size=world_size,
        rank=rank,
        block_size=block_size,
        block_ranges=tuple(local_block_ranges),
        sample_indices=tuple(local_sample_indices),
        real_count=len(local_sample_indices),
        gather_count=max(counts, default=0),
    )


def build_shard_spec(
    total_size: int,
    world_size: int,
    rank: int,
    block_size: int | None = None,
) -> ShardSpec:
    """Build an explicit shard description.

    When ``block_size`` is ``None``, the helper preserves the legacy contiguous
    split while still materializing explicit indices. When ``block_size`` is a
    positive integer, blocks of that size are assigned to ranks in block-cyclic
    order.
    """
    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within [0, world_size)")
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive when provided")

    if block_size is None:
        return _build_contiguous_shard_spec(total_size, world_size, rank)
    return _build_block_cyclic_shard_spec(total_size, world_size, rank, block_size)


def build_sharded_dataset_view(dataset: Any, shard: ShardSpec) -> Subset:
    """Return a local dataset view for the shard."""
    return Subset(dataset, list(shard.sample_indices))


def build_shard_position_tensor(
    shard: ShardSpec,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the local sample positions as an ``int64`` tensor."""
    return torch.tensor(shard.sample_indices, device=device, dtype=torch.int64)


def _is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _resolve_trailing_shape(
    local_tensor: torch.Tensor,
    counts: list[int] | None = None,
) -> tuple[int, ...]:
    """Resolve the common trailing shape without reducing shape metadata.

    Empty local shards may not know the embedding width, so the shape from
    non-empty ranks is authoritative. The implementation uses all-gather rather
    than all-reduce because shape metadata is not a numeric statistic and reduce
    operations on int64 metadata can produce invalid packed values on some
    distributed backends.
    """
    local_shape = tuple(int(dim) for dim in local_tensor.shape[1:])
    if not _is_distributed_initialized():
        return local_shape

    world_size = dist.get_world_size()
    if counts is not None and len(counts) != world_size:
        raise ValueError("counts length must match distributed world size")

    local_rank_tensor = torch.tensor([len(local_shape)], device=local_tensor.device, dtype=torch.int64)
    gathered_rank_tensors = [torch.zeros_like(local_rank_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_rank_tensors, local_rank_tensor)
    trailing_ranks = [int(item.item()) for item in gathered_rank_tensors]
    max_trailing_rank = max(trailing_ranks, default=0)

    if max_trailing_rank == 0:
        return ()

    local_shape_tensor = torch.full(
        (max_trailing_rank,),
        -1,
        device=local_tensor.device,
        dtype=torch.int64,
    )
    if local_shape:
        local_shape_tensor[: len(local_shape)] = torch.tensor(
            local_shape,
            device=local_tensor.device,
            dtype=torch.int64,
        )

    gathered_shape_tensors = [torch.empty_like(local_shape_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_shape_tensors, local_shape_tensor)
    gathered_shapes = [
        tuple(int(value) for value in shape_tensor[: trailing_ranks[rank]].tolist())
        for rank, shape_tensor in enumerate(gathered_shape_tensors)
    ]

    if counts is None:
        authoritative_shapes = gathered_shapes
    else:
        authoritative_shapes = [
            shape for rank, shape in enumerate(gathered_shapes) if counts[rank] > 0
        ]

    if not authoritative_shapes:
        return local_shape

    expected_shape = authoritative_shapes[0]
    mismatched_shapes = [shape for shape in authoritative_shapes if shape != expected_shape]
    if mismatched_shapes:
        raise ValueError(
            "non-empty ranks produced incompatible trailing shapes: "
            f"{authoritative_shapes}"
        )

    return expected_shape


def _pad_first_dim(
    local_tensor: torch.Tensor,
    target_count: int,
    trailing_shape: tuple[int, ...],
) -> torch.Tensor:
    if local_tensor.ndim == 0:
        raise ValueError("local_tensor must have at least one dimension")

    expected_shape = (local_tensor.shape[0], *trailing_shape)
    if local_tensor.shape != expected_shape:
        if local_tensor.shape[0] == 0:
            local_tensor = local_tensor.new_empty(expected_shape)
        else:
            raise ValueError(
                "local_tensor trailing dimensions do not match gathered trailing shape: "
                f"{local_tensor.shape[1:]} vs {trailing_shape}"
            )

    if target_count == local_tensor.shape[0]:
        return local_tensor.contiguous()

    padded = local_tensor.new_zeros((target_count, *trailing_shape))
    if local_tensor.shape[0] > 0:
        padded[: local_tensor.shape[0]].copy_(local_tensor)
    return padded.contiguous()


@torch.no_grad()
def gather_tensor_first_dim(local_tensor: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Gather a tensor across ranks by explicitly exchanging local counts first."""
    if local_tensor.ndim == 0:
        raise ValueError("local_tensor must have at least one dimension")

    if not _is_distributed_initialized():
        return local_tensor.contiguous(), [int(local_tensor.shape[0])]

    world_size = dist.get_world_size()
    local_count = torch.tensor([local_tensor.shape[0]], device=local_tensor.device, dtype=torch.int64)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(item.item()) for item in gathered_counts]

    trailing_shape = _resolve_trailing_shape(local_tensor, counts)

    max_count = max(counts, default=0)
    if max_count == 0:
        empty = local_tensor.new_empty((0, *trailing_shape))
        return empty, counts

    padded = _pad_first_dim(local_tensor, max_count, trailing_shape)
    gathered = local_tensor.new_empty((world_size * max_count, *trailing_shape), dtype=padded.dtype)
    dist.all_gather_into_tensor(gathered, padded)
    return gathered, counts


def _trim_rank_order_gathered_tensor(
    gathered: torch.Tensor,
    counts: list[int],
    total_size: int,
) -> torch.Tensor:
    """Trim gathered rank-major data down to its real samples."""
    if gathered.ndim == 0:
        raise ValueError("gathered tensor must have at least one dimension")
    if total_size < 0:
        raise ValueError("total_size must be non-negative")

    world_size = len(counts)
    if world_size == 0:
        if total_size != 0:
            raise ValueError("counts cannot be empty when total_size is non-zero")
        return gathered[:0].contiguous()

    if any(count < 0 for count in counts):
        raise ValueError("counts must be non-negative")
    if sum(counts) != total_size:
        raise ValueError(f"counts sum {sum(counts)} does not match total_size {total_size}")

    if gathered.shape[0] == 0:
        if total_size != 0:
            raise ValueError("gathered tensor is empty but total_size is non-zero")
        return gathered.contiguous()

    if gathered.shape[0] % world_size != 0:
        raise ValueError("gathered first dimension must be divisible by number of counts")

    gather_count = gathered.shape[0] // world_size
    if any(count > gather_count for count in counts):
        raise ValueError("counts cannot exceed gathered per-rank padding")

    pieces = gathered.reshape(world_size, gather_count, *gathered.shape[1:])
    trimmed_parts = [pieces[rank, :count] for rank, count in enumerate(counts) if count > 0]
    trimmed = torch.cat(trimmed_parts, dim=0) if trimmed_parts else gathered[:0]

    if trimmed.shape[0] != total_size:
        raise ValueError(
            f"trimmed tensor first dimension {trimmed.shape[0]} does not match total_size {total_size}"
        )
    return trimmed.contiguous()


def trim_gathered_tensor(
    gathered: torch.Tensor,
    counts: list[int],
    total_size: int,
) -> torch.Tensor:
    """Trim a gathered tensor back to its real samples in rank order."""
    return _trim_rank_order_gathered_tensor(gathered, counts=counts, total_size=total_size)


def reorder_gathered_tensor(
    gathered: torch.Tensor,
    positions: torch.Tensor,
    counts: list[int],
    total_size: int,
) -> torch.Tensor:
    """Trim rank-major gathered data and restore the original global order."""
    trimmed = _trim_rank_order_gathered_tensor(gathered, counts=counts, total_size=total_size)
    trimmed_positions = _trim_rank_order_gathered_tensor(
        positions,
        counts=counts,
        total_size=total_size,
    ).to(dtype=torch.int64)

    if trimmed_positions.ndim != 1:
        raise ValueError("positions must be a 1D tensor after trimming")
    if total_size == 0:
        return trimmed.contiguous()

    min_position = int(trimmed_positions.min().item())
    max_position = int(trimmed_positions.max().item())
    if min_position < 0 or max_position >= total_size:
        raise ValueError("positions must stay within [0, total_size)")

    unique_positions = torch.unique(trimmed_positions)
    if unique_positions.numel() != total_size:
        raise ValueError("positions must be unique and cover total_size entries")

    sorted_positions = torch.sort(trimmed_positions).values
    expected_positions = torch.arange(total_size, device=trimmed_positions.device, dtype=torch.int64)
    if not torch.equal(sorted_positions, expected_positions):
        raise ValueError("positions must cover the full [0, total_size) range")

    reordered = gathered.new_empty((total_size, *gathered.shape[1:]))
    reordered[trimmed_positions] = trimmed
    return reordered.contiguous()


__all__ = [
    "ShardSpec",
    "build_shard_position_tensor",
    "build_shard_spec",
    "build_sharded_dataset_view",
    "gather_tensor_first_dim",
    "reorder_gathered_tensor",
    "trim_gathered_tensor",
]
