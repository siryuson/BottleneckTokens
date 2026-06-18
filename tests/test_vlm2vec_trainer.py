from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vlm2emb.training.trainers.vlm2vec_trainer import VLM2VecTrainer


class _Model:
    def train(self) -> None:
        pass


def _make_trainer(*, gc_sort_by_length: bool) -> VLM2VecTrainer:
    trainer = object.__new__(VLM2VecTrainer)
    trainer.args = SimpleNamespace(temperature=0.02)
    trainer.gc_chunk_size = 2
    trainer.gc_sort_by_length = gc_sort_by_length
    trainer.accelerator = SimpleNamespace(num_processes=1)
    trainer.processing_class = SimpleNamespace()
    return trainer


def test_vlm2vec_input_token_length_prefers_attention_mask() -> None:
    sample = {
        "input_ids": torch.ones((1, 5), dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long),
    }

    assert VLM2VecTrainer._input_token_length(sample) == 2


def test_vlm2vec_input_token_length_falls_back_to_input_ids() -> None:
    sample = {"input_ids": torch.ones((1, 5), dtype=torch.long)}

    assert VLM2VecTrainer._input_token_length(sample) == 5


def test_vlm2vec_training_step_passes_query_sort_key_without_reordering(
    monkeypatch,
) -> None:
    trainer = _make_trainer(gc_sort_by_length=True)
    queries = [
        {
            "id": "q0",
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        },
        {
            "id": "q1",
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
    ]
    positives = [
        {"id": "p0", "input_ids": torch.ones((1, 1), dtype=torch.long)},
        {"id": "p1", "input_ids": torch.ones((1, 9), dtype=torch.long)},
    ]
    captured: dict[str, Any] = {}

    def fake_grad_cache_accumulate(**kwargs):
        captured.update(kwargs)
        return torch.tensor(0.0), [[], []]

    monkeypatch.setattr(
        "vlm2emb.training.grad_cache.grad_cache_accumulate",
        fake_grad_cache_accumulate,
    )

    loss = trainer.training_step(
        _Model(),
        {"query": queries, "positive": positives},
    )

    assert loss.item() == 0.0
    assert captured["inputs"] == [queries, positives]
    assert [sample["id"] for sample in queries] == ["q0", "q1"]
    assert [sample["id"] for sample in positives] == ["p0", "p1"]
    assert captured["aligned_sort_group_idx"] == 0
    assert captured["aligned_sort_key_fn"](queries[0]) == 5
    assert captured["aligned_sort_key_fn"](queries[1]) == 2
    assert captured["aligned_sort_key_fn"] is not None


def test_vlm2vec_training_step_omits_sort_key_when_disabled(monkeypatch) -> None:
    trainer = _make_trainer(gc_sort_by_length=False)
    captured: dict[str, Any] = {}

    def fake_grad_cache_accumulate(**kwargs):
        captured.update(kwargs)
        return torch.tensor(0.0), [[], []]

    monkeypatch.setattr(
        "vlm2emb.training.grad_cache.grad_cache_accumulate",
        fake_grad_cache_accumulate,
    )

    trainer.training_step(
        _Model(),
        {
            "query": [{"input_ids": torch.ones((1, 5), dtype=torch.long)}],
            "positive": [{"input_ids": torch.ones((1, 7), dtype=torch.long)}],
        },
    )

    assert captured["aligned_sort_key_fn"] is None
    assert captured["aligned_sort_group_idx"] == 0
