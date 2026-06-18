import math
from collections.abc import Iterator, Sized
from typing import cast

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

__all__ = ["DistributedSampler", "ChunkSampler"]


# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_sampler.py
class DistributedSampler(Sampler[int]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        consumed_samples=0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.consumed_indicies = consumed_samples // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        # skip consumed_samples
        indices = indices[self.consumed_indicies :]
        assert len(indices) == self.num_samples - self.consumed_indicies

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.consumed_indicies

    def set_epoch(self, epoch: int, consumed_samples=0) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.consumed_indicies = consumed_samples // self.num_replicas


class ChunkSampler(Sampler[int]):
    """Sampler that yields indices in chunks, with chunks randomly shuffled.

    Each chunk starts at a multiple of `chunk_size` and contains `chunk_size`
    consecutive indices. Chunks are randomly shuffled, but indices within
    each chunk remain in order.

    This is useful for datasets where consecutive samples should stay together
    (e.g., samples from the same source dataset in interleaved datasets).

    Args:
        data_source: Dataset to sample from.
        chunk_size: Size of each contiguous chunk, usually interleave_batch_size.
        shuffle: Whether to shuffle chunk order between epochs.
        seed: Random seed for reproducible shuffling.
        drop_last: Whether to drop the final incomplete chunk.

    Example:
        >>> dataset = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> sampler = ChunkSampler(dataset, chunk_size=3, shuffle=True)
        >>> list(sampler)  # e.g., [6, 7, 8, 0, 1, 2, 3, 4, 5, 9] (chunks shuffled)
        >>>                # chunk2=[6,7,8], chunk0=[0,1,2], chunk1=[3,4,5], chunk3=[9]
    """

    def __init__(
        self,
        data_source: Dataset,
        chunk_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.data_source = data_source
        sized_data_source = cast(Sized, data_source)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate number of chunks
        self.num_chunks = len(sized_data_source) // chunk_size
        self.remainder = len(sized_data_source) % chunk_size

        if self.drop_last:
            self.num_samples = self.num_chunks * chunk_size
        else:
            self.num_samples = len(sized_data_source)

    def __iter__(self) -> Iterator[int]:
        # Generate chunk indices [0, 1, 2, ..., num_chunks-1]
        chunk_indices = list(range(self.num_chunks))

        # Shuffle chunk order deterministically
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            chunk_indices = torch.randperm(self.num_chunks, generator=g).tolist()

        # Expand chunks to sample indices
        indices = []
        for chunk_idx in chunk_indices:
            start = chunk_idx * self.chunk_size
            end = start + self.chunk_size
            indices.extend(range(start, end))

        # Handle remainder (last incomplete chunk)
        if not self.drop_last and self.remainder > 0:
            start = self.num_chunks * self.chunk_size
            indices.extend(range(start, len(cast(Sized, self.data_source))))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across epochs.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch
