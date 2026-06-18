"""Sampler implementations for retrieval training.

This module provides custom samplers for training, integrated with HuggingFace Trainer
via `_get_train_sampler()` override.

Available samplers:
    - BatchInterleaveSampler: Batch-wise interleaving for multi-source datasets
"""

from vlm2emb.training.samplers.batch_interleave import BatchInterleaveSampler

__all__ = [
    "BatchInterleaveSampler",
]
