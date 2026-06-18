"""Distributed helpers for evaluation protocols."""

from .retrieval_protocol import (
    ShardSpec,
    build_shard_position_tensor,
    build_shard_spec,
    build_sharded_dataset_view,
    gather_tensor_first_dim,
    reorder_gathered_tensor,
    trim_gathered_tensor,
)

__all__ = [
    "ShardSpec",
    "build_shard_position_tensor",
    "build_shard_spec",
    "build_sharded_dataset_view",
    "gather_tensor_first_dim",
    "reorder_gathered_tensor",
    "trim_gathered_tensor",
]
