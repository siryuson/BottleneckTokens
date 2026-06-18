from torch.utils.data import Dataset

from vlm2emb.training.samplers import BatchInterleaveSampler


class DummyCombinedDataset(Dataset):
    def __init__(self, lengths: list[int], weights: list[float] | None = None) -> None:
        self.dataset_lengths = lengths
        self.dataset_offsets = self._compute_offsets(lengths)
        self.weights = weights if weights is not None else [1.0] * len(lengths)

    @staticmethod
    def _compute_offsets(lengths: list[int]) -> list[int]:
        offsets = [0]
        for length in lengths[:-1]:
            offsets.append(offsets[-1] + length)
        return offsets

    def __len__(self) -> int:
        return sum(self.dataset_lengths)

    def __getitem__(self, idx: int) -> int:
        return idx


def test_sequential_within_dataset_preserves_local_order() -> None:
    dataset = DummyCombinedDataset([3, 2])
    sampler = BatchInterleaveSampler(
        dataset=dataset,
        batch_size=1,
        stopping_strategy="first_exhausted",
        shuffle_within_dataset=False,
        seed=42,
    )

    assert list(sampler) == [0, 3, 1, 4, 2]


def test_legacy_shuffle_alias_still_controls_within_dataset_order() -> None:
    dataset = DummyCombinedDataset([3, 2])
    sampler = BatchInterleaveSampler(
        dataset=dataset,
        batch_size=1,
        stopping_strategy="first_exhausted",
        shuffle=False,
        seed=42,
    )

    assert list(sampler) == [0, 3, 1, 4, 2]


def test_set_epoch_keeps_sequential_order_when_shuffle_disabled() -> None:
    dataset = DummyCombinedDataset([4, 4])
    sampler = BatchInterleaveSampler(
        dataset=dataset,
        batch_size=1,
        stopping_strategy="first_exhausted",
        shuffle_within_dataset=False,
        seed=7,
    )

    epoch0 = list(sampler)
    sampler.set_epoch(1)

    assert list(sampler) == epoch0


def test_set_epoch_changes_within_dataset_order_when_shuffle_enabled() -> None:
    dataset = DummyCombinedDataset([4, 4])
    sampler = BatchInterleaveSampler(
        dataset=dataset,
        batch_size=1,
        stopping_strategy="first_exhausted",
        shuffle_within_dataset=True,
        seed=7,
    )

    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert epoch0 != epoch1
    assert sorted(epoch0) == sorted(epoch1) == list(range(8))
