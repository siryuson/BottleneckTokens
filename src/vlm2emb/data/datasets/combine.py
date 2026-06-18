"""Combined dataset for multi-source training.


This module provides CombinedDataset, which combines multiple PyTorch Datasets
and stores metadata for BatchInterleaveSampler.

Design
    - CombinedDataset stores metadata (lengths, offsets, weights) as attributes
    - BatchInterleaveSampler (in training/samplers/) reads this metadata
    - Distributed training handled automatically by accelerator.prepare()

Usage
    >>> from vlm2emb.data.datasets import CombinedDataset
    >>> dataset = CombinedDataset(
    ...     datasets=[ds1, ds2, ds3],
    ...     weights=[1.0, 2.0, 1.0],
    ...     names=["mmeb", "vidore", "visrag"],
    ... )
    >>> # In Trainer, BatchInterleaveSampler reads dataset.dataset_lengths, etc.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sized
from typing import Any, cast

from torch.utils.data import Dataset

from vlm2emb.auto import AutoDataset
from vlm2emb.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "CombinedDataset",
    "load_combined_dataset",
]


class CombinedDataset(Dataset):
    """Combined dataset for multi-source training with metadata.


    Combines multiple PyTorch Datasets and stores metadata attributes that
    BatchInterleaveSampler uses for batch-wise interleaving.

    Attributes:
        datasets: List of sub-datasets.
        dataset_lengths: Length of each sub-dataset.
        dataset_offsets: Starting offset of each sub-dataset in concatenated index space.
        weights: Sampling weights for each sub-dataset.
        names: Names of each sub-dataset (for logging).

    Args:
        datasets: List of PyTorch Datasets to combine.
        weights: Sampling weights for each dataset. Default: equal weights.
        names: Names for each dataset. Default: dataset_0, dataset_1, ...

    Example
        >>> ds1 = AutoDataset.build("mmeb_train", path="/data/mmeb", subset="ImageNet_1K")
        >>> ds2 = AutoDataset.build("vidore_train", path="/data/vidore")
        >>> combined = CombinedDataset(
        ...     datasets=[ds1, ds2],
        ...     weights=[2.0, 1.0],
        ...     names=["mmeb", "vidore"],
        ... )
        >>> print(len(combined))  # Total samples
        >>> print(combined.dataset_lengths)  # [len(ds1), len(ds2)]
        >>> sample = combined[0]  # Access by global index
    """

    def __init__(
        self,
        datasets: list[Dataset[Any]],
        weights: list[float] | None = None,
        names: list[str] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("datasets cannot be empty")

        self.datasets = datasets
        self.num_datasets = len(datasets)

        # Compute metadata
        self.dataset_lengths = [len(cast(Sized, ds)) for ds in datasets]
        self.dataset_offsets = self._compute_offsets(self.dataset_lengths)
        self.weights = weights if weights is not None else [1.0] * self.num_datasets
        self.names = names if names is not None else [f"dataset_{i}" for i in range(self.num_datasets)]

        # Validate
        if len(self.weights) != self.num_datasets:
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"number of datasets ({self.num_datasets})"
            )
        if len(self.names) != self.num_datasets:
            raise ValueError(
                f"names length ({len(self.names)}) must match "
                f"number of datasets ({self.num_datasets})"
            )

        # Total length
        self._total_length = sum(self.dataset_lengths)

        logger.info(
            f"CombinedDataset created: "
            f"num_datasets={self.num_datasets}, "
            f"total_length={self._total_length}, "
            f"lengths={self.dataset_lengths}, "
            f"names={self.names}"
        )

    @staticmethod
    def _compute_offsets(lengths: list[int]) -> list[int]:
        """Compute cumulative offsets from lengths."""
        offsets = [0]
        for length in lengths[:-1]:
            offsets.append(offsets[-1] + length)
        return offsets

    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self._total_length

    def _get_dataset_and_local_index(self, global_idx: int) -> tuple[int, int]:
        """Convert global index to (dataset_index, local_index).


        Args:
            global_idx: Index in concatenated space.

        Returns:
            Tuple of (dataset_index, local_index_within_dataset).
        """
        if global_idx < 0 or global_idx >= self._total_length:
            raise IndexError(f"Index {global_idx} out of range [0, {self._total_length})")

        for i, (offset, length) in enumerate(
            zip(self.dataset_offsets, self.dataset_lengths, strict=True)
        ):
            if global_idx < offset + length:
                return i, global_idx - offset

        # Should not reach here
        raise IndexError(f"Index {global_idx} out of range")

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index.


        Args:
            idx: Global index in concatenated space.

        Returns:
            Sample from the appropriate sub-dataset.
        """
        ds_idx, local_idx = self._get_dataset_and_local_index(idx)
        return self.datasets[ds_idx][local_idx]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Batch fetch samples by global indices.


        Args:
            indices: List of global indices.

        Returns:
            List of samples.
        """
        if not indices:
            return []

        # Group local indices by sub-dataset so child datasets can use their
        # own batch-optimized __getitems__ implementations.
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_pos, global_idx in enumerate(indices):
            ds_idx, local_idx = self._get_dataset_and_local_index(global_idx)
            grouped.setdefault(ds_idx, []).append((output_pos, local_idx))

        results: list[Any] = [None] * len(indices)

        for ds_idx, members in grouped.items():
            dataset = self.datasets[ds_idx]
            local_indices = [local_idx for _, local_idx in members]
            batch_getitems = getattr(dataset, "__getitems__", None)

            if callable(batch_getitems):
                fetched = list(batch_getitems(local_indices))
            else:
                fetched = [dataset[local_idx] for local_idx in local_indices]

            for (output_pos, _), item in zip(members, fetched, strict=True):
                results[output_pos] = item

        return results

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all samples in order.

        """
        for ds in self.datasets:
            yield from cast(Iterator[Any], ds)


@AutoDataset.register("combined")
def load_combined_dataset(
    datasets: Config,
    **kwargs: Any,
) -> CombinedDataset:
    """Load and combine multiple datasets from config.


    Args:
        datasets: Dataset config dict; each sub-config must contain a ``type`` field.
        **kwargs: Unexpected parameters raise TypeError.

    Returns:
        CombinedDataset instance.

    Example config
        ```yaml
        type: combined
        datasets:
          mmeb:
            type: mmeb_train
            path: /path/to/mmeb
            subset: ImageNet_1K
            weight: 2.0
          vidore:
            type: vidore_train
            path: /path/to/vidore
            weight: 1.0
        ```

    Note:
        - If weight is specified in config, it will be converted to probabilities
        - probabilities parameter takes precedence over weight if both are specified
    """
    if kwargs:
        unknown_keys = sorted(kwargs.keys())
        raise TypeError(f"load_combined_dataset() got unexpected keyword argument(s): {unknown_keys}")

    built_datasets: list[Dataset] = []
    weights: list[float] = []
    names: list[str] = []

    for name, dataset_config in datasets.items():
        if str(name).startswith("_"):
            continue
        if not isinstance(dataset_config, Mapping):
            raise ValueError(f"Dataset config '{name}' must be a mapping, got {type(dataset_config)}")
        if "type" not in dataset_config:
            raise ValueError(f"Dataset config '{name}' must contain 'type' field")

        # Copy config to avoid modifying original
        config_copy = dict(dataset_config)
        dataset_type = config_copy.get("type")
        weight = config_copy.pop("weight", 1.0)
        weights.append(weight)
        names.append(name)

        ds = cast(Dataset[Any], AutoDataset.from_config(config_copy))
        built_datasets.append(ds)
        logger.info(
            "Loaded dataset '%s' (type=%s, weight=%s, len=%s)",
            name,
            dataset_type,
            weight,
            len(cast(Sized, ds)),
        )

    return CombinedDataset(
        datasets=built_datasets,
        weights=weights,
        names=names,
    )
