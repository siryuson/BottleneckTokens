"""Batch interleave sampler for multi-source dataset training.


This sampler generates indices such that each batch contains samples from a single
data source, while batches alternate between sources based on weights.


Design
    - Pre-generates the complete index sequence for consistency
    - __len__ returns the exact length of the sequence
    - __iter__ iterates over the pre-generated sequence
    - Supports checkpoint recovery via set_epoch()

    - __len__ 
    - __iter__ 
    -  set_epoch() 

Integration
    Trainer._get_train_sampler() returns this sampler when dataset has metadata.
    Trainer._get_train_sampler() 
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from vlm2emb.auto import AutoSampler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["BatchInterleaveSampler"]

StoppingStrategy = Literal["all_exhausted", "first_exhausted"]


class _InterleaveDataset(Protocol):
    dataset_lengths: list[int]
    dataset_offsets: list[int]
    weights: list[float]


@AutoSampler.register("batch_interleave")
class BatchInterleaveSampler(Sampler[int]):
    """Batch-wise interleaving sampler for multi-source datasets.


    Pre-generates the complete index sequence to ensure __len__ and __iter__ are consistent.
     __len__  __iter__ 

    Sampling Strategy
        - Weighted: When weights are not None and not all equal to 1
        - Round-robin: Otherwise

        -  weights  None  1 

    Stopping Strategy
        - all_exhausted: All datasets must be exhausted at least once before stopping.
                         Short datasets are reset and resampled until all have been
                         fully sampled at least once.
        - first_exhausted: Stop when any dataset is exhausted.

        - all_exhausted
        - first_exhausted

    Args:
        dataset: CombinedDataset with metadata attributes:
                 - dataset_lengths: list[int] - length of each sub-dataset
                 - dataset_offsets: list[int] - starting offset of each sub-dataset
                 - weights: list[float] - sampling weights for each sub-dataset
        batch_size: Number of consecutive samples from each source.
        stopping_strategy: When to stop iteration ("all_exhausted" or "first_exhausted").
        shuffle_within_dataset: Whether to shuffle indices within each dataset.
        seed: Random seed for reproducibility.

    Example
        >>> sampler = BatchInterleaveSampler(
        ...     dataset=combined_dataset,
        ...     batch_size=32,
        ...     stopping_strategy="all_exhausted",
        ...     shuffle_within_dataset=True,
        ...     seed=42,
        ... )
        >>> len(sampler)  # Exact number of indices
        >>> list(sampler)  # Pre-generated index sequence
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        stopping_strategy: StoppingStrategy = "all_exhausted",
        shuffle_within_dataset: bool | None = None,
        seed: int = 0,
        per_device_batch_size: int | None = None,
        num_processes: int | None = None,
        shuffle: bool | None = None,
    ) -> None:
        # Read metadata from dataset
        if not hasattr(dataset, "dataset_lengths"):
            raise ValueError(
                "Dataset must have 'dataset_lengths' attribute. "
                "Use CombinedDataset or ensure dataset has metadata."
            )

        interleave_dataset = cast(_InterleaveDataset, dataset)
        self.lengths: list[int] = list(interleave_dataset.dataset_lengths)
        self.offsets: list[int] = list(interleave_dataset.dataset_offsets)
        self.weights: list[float] = list(
            getattr(interleave_dataset, "weights", [1.0] * len(self.lengths))
        )

        self.batch_size = batch_size
        self.stopping_strategy = stopping_strategy
        if shuffle_within_dataset is None:
            shuffle_within_dataset = True if shuffle is None else shuffle
        elif shuffle is not None and shuffle != shuffle_within_dataset:
            raise ValueError(
                "Received conflicting values for 'shuffle_within_dataset' and "
                "legacy alias 'shuffle'."
            )
        self.shuffle_within_dataset = shuffle_within_dataset
        self.seed = seed
        self.epoch = 0
        self._per_device_batch_size = per_device_batch_size
        self._num_processes = num_processes

        # Validate interleave_batch_size is divisible by per_device_batch_size
        if per_device_batch_size is not None and batch_size % per_device_batch_size != 0:
            raise ValueError(
                f"interleave_batch_size ({batch_size}) must be divisible by "
                f"per_device_batch_size ({per_device_batch_size})"
            )

        self.num_datasets = len(self.lengths)

        # Determine sampling mode
        self._use_weighted = (
            self.weights is not None
            and len(self.weights) == self.num_datasets
            and not all(w == 1.0 for w in self.weights)
        )

        if self._use_weighted:
            total_weight = sum(self.weights)
            self.probabilities = [w / total_weight for w in self.weights]
        else:
            self.probabilities = [1.0 / self.num_datasets] * self.num_datasets

        # Pre-generate index sequence
        self._indices: list[int] = []
        self._generate_indices()

        logger.info(
            f"BatchInterleaveSampler initialized: "
            f"num_datasets={self.num_datasets}, "
            f"batch_size={batch_size}, "
            f"stopping_strategy={stopping_strategy}, "
            f"lengths={self.lengths}, "
            f"mode={'weighted' if self._use_weighted else 'round_robin'}, "
            f"shuffle_within_dataset={self.shuffle_within_dataset}, "
            f"stride_interleave={'enabled' if per_device_batch_size and num_processes else 'disabled'}, "
            f"total_indices={len(self._indices)}"
        )

    def _should_stop(self, exhausted_count: list[int]) -> bool:
        """Check if sampling should stop based on stopping strategy."""
        if self.stopping_strategy == "first_exhausted":
            return any(count > 0 for count in exhausted_count)
        # all_exhausted: stop when ALL datasets have been exhausted at least once
        return all(count > 0 for count in exhausted_count)

    def _init_dataset_indices(self, generator: torch.Generator) -> list[list[int]]:
        """Initialize indices for each dataset."""
        dataset_indices: list[list[int]] = []
        for ds_idx in range(self.num_datasets):
            length = self.lengths[ds_idx]
            offset = self.offsets[ds_idx]
            if self.shuffle_within_dataset:
                indices = torch.randperm(length, generator=generator).tolist()
            else:
                indices = list(range(length))
            dataset_indices.append([idx + offset for idx in indices])
        return dataset_indices

    def _reshuffle_dataset(
        self,
        ds_idx: int,
        dataset_indices: list[list[int]],
        generator: torch.Generator,
    ) -> None:
        length = self.lengths[ds_idx]
        offset = self.offsets[ds_idx]
        if self.shuffle_within_dataset:
            new_indices = torch.randperm(length, generator=generator).tolist()
        else:
            new_indices = list(range(length))
        dataset_indices[ds_idx] = [idx + offset for idx in new_indices]

    def _select_dataset(self, rng: np.random.Generator, rr_idx: int) -> int:
        """Select next dataset based on sampling strategy."""
        if self._use_weighted:
            return int(rng.choice(self.num_datasets, p=self.probabilities))
        return rr_idx % self.num_datasets

    def _sample_batch_from_dataset(
        self,
        ds_idx: int,
        positions: list[int],
        exhausted_count: list[int],
        dataset_indices: list[list[int]],
        generator: torch.Generator,
    ) -> list[int]:
        """Sample batch_size indices from a single dataset.

        Returns the sampled indices. Handles dataset exhaustion and reset.
        """
        batch_indices: list[int] = []

        for _ in range(self.batch_size):
            # Handle dataset exhaustion
            if positions[ds_idx] >= self.lengths[ds_idx]:
                exhausted_count[ds_idx] += 1

                if self._should_stop(exhausted_count):
                    break

                # Reset and optionally reshuffle for next round
                positions[ds_idx] = 0
                self._reshuffle_dataset(ds_idx, dataset_indices, generator)

            # Sample index
            if positions[ds_idx] < self.lengths[ds_idx]:
                batch_indices.append(dataset_indices[ds_idx][positions[ds_idx]])
                positions[ds_idx] += 1

        return batch_indices

    def _generate_indices(self) -> None:
        """Generate the complete index sequence."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        rng = np.random.default_rng(self.seed + self.epoch)

        dataset_indices = self._init_dataset_indices(generator)
        positions = [0] * self.num_datasets
        exhausted_count = [0] * self.num_datasets

        self._indices = []
        rr_idx = 0

        while not self._should_stop(exhausted_count):
            ds_idx = self._select_dataset(rng, rr_idx)
            rr_idx += 1

            batch = self._sample_batch_from_dataset(
                ds_idx, positions, exhausted_count, dataset_indices, generator
            )
            self._indices.extend(batch)

        # Apply stride interleave for distributed load balancing
        self._apply_stride_interleave()

    def _apply_stride_interleave(self) -> None:
        """Rearrange indices for round-robin distribution across GPUs.

         GPU 

        Example
            Input:  [0, 1, 2, 3, ..., 1023] (step_size = 128 * 8 = 1024)
            Output: [0, 8, 16, 24, ..., 504, 1, 9, 17, ..., 505, 2, 10, ...]

            Each GPU gets indices in round-robin order:
            - GPU 0: 0, 8, 16, ...
            - GPU 1: 1, 9, 17, ...
            - ...
        """
        if self._per_device_batch_size is None or self._num_processes is None:
            return
        if self._num_processes <= 1:
            return

        pbs = self._per_device_batch_size
        step_size = pbs * self._num_processes

        result: list[int] = []
        total = len(self._indices)

        for pos in range(0, total, step_size):
            step_indices = self._indices[pos : pos + step_size]

            if len(step_indices) < step_size:
                # Incomplete step, just append as-is
                result.extend(step_indices)
                continue

            # Reshape to (num_processes, per_device_batch_size)
            # Then transpose to (per_device_batch_size, num_processes)
            # Then flatten
            for batch_offset in range(pbs):
                for gpu_idx in range(self._num_processes):
                    src_idx = gpu_idx * pbs + batch_offset
                    result.append(step_indices[src_idx])

        self._indices = result

    def __len__(self) -> int:
        """Return the exact number of indices in the sequence."""
        return len(self._indices)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the pre-generated index sequence."""
        return iter(self._indices)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across epochs.

         epoch  epoch 

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch
        self._generate_indices()

    def state_dict(self) -> dict:
        """Return state for checkpoint recovery."""
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "lengths": self.lengths,
            "offsets": self.offsets,
            "weights": self.weights,
            "batch_size": self.batch_size,
            "stopping_strategy": self.stopping_strategy,
            "shuffle_within_dataset": self.shuffle_within_dataset,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.epoch = state["epoch"]
        self.seed = state["seed"]
        self.shuffle_within_dataset = state.get(
            "shuffle_within_dataset", self.shuffle_within_dataset
        )
        if state["lengths"] != self.lengths:
            logger.warning(
                f"Dataset lengths mismatch: saved={state['lengths']}, "
                f"current={self.lengths}"
            )
        self._generate_indices()
