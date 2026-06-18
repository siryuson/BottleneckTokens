from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import DynamicCache

from vlm2emb.inference.btoks import generate_from_btoks_cache


def _make_cache(seq_len: int, batch_size: int = 1) -> DynamicCache:
    cache = DynamicCache()
    key_states = torch.arange(batch_size * seq_len, dtype=torch.float32).view(
        batch_size, 1, seq_len, 1,
    )
    value_states = key_states + 100
    cache.update(key_states, value_states, layer_idx=0)
    return cache


class FakeBackboneModel:
    def __init__(self, next_token_ids: list[int], eos_token_id: int):
        self.calls: list[dict[str, torch.Tensor | DynamicCache | bool]] = []
        self.next_token_ids = next_token_ids
        self.eos_token_id = eos_token_id

    def get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size, seq_len = input_ids.shape
        base = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len)
        return base.expand(3, batch_size, seq_len), None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        step = len(self.calls) - 1
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        vocab_size = 16
        token_id = self.next_token_ids[min(step, len(self.next_token_ids) - 1)]

        logits = torch.full((batch_size, seq_len, vocab_size), -1000.0)
        logits[:, -1, token_id] = 0.0

        cache_len = kwargs["past_key_values"].get_seq_length() + seq_len
        return SimpleNamespace(
            logits=logits,
            past_key_values=_make_cache(cache_len, batch_size=batch_size),
        )


class FakeTopLevelBackboneModel:
    def __init__(self, next_token_ids: list[int], eos_token_id: int):
        self.inner = FakeBackboneModel(next_token_ids=next_token_ids, eos_token_id=eos_token_id)
        self.model = SimpleNamespace(get_rope_index=self.inner.get_rope_index)
        self.calls = self.inner.calls

    def __call__(self, **kwargs):
        return self.inner(**kwargs)


def test_generate_from_btoks_cache_uses_btoks_prefix_and_decodes_tokens():
    eos_token_id = 2
    backbone = FakeBackboneModel(next_token_ids=[7, eos_token_id], eos_token_id=eos_token_id)
    encode_out = {
        "input_ids": torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        "btoks_token_mask": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool),
        "past_key_values": _make_cache(seq_len=5),
    }
    prompt_batch = {
        "input_ids": torch.tensor([[20, 21]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }

    generated = generate_from_btoks_cache(
        encode_out=encode_out,
        backbone_model=backbone,
        prompt_batch=prompt_batch,
        eos_token_id=eos_token_id,
        max_new_tokens=4,
    )

    assert generated.tolist() == [[7, eos_token_id]]

    first_call = backbone.calls[0]
    assert first_call["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert first_call["cache_position"].tolist() == [2, 3]
    assert first_call["position_ids"].tolist() == [[[5, 6]], [[5, 6]], [[5, 6]]]

    second_call = backbone.calls[1]
    assert second_call["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert second_call["cache_position"].tolist() == [4]
    assert second_call["position_ids"].tolist() == [[[7]], [[7]], [[7]]]


def test_generate_from_btoks_cache_uses_inner_qwen_rope_index():
    eos_token_id = 2
    backbone = FakeTopLevelBackboneModel(next_token_ids=[eos_token_id], eos_token_id=eos_token_id)
    encode_out = {
        "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "btoks_token_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
        "past_key_values": _make_cache(seq_len=3),
    }
    prompt_batch = {
        "input_ids": torch.tensor([[20, 21]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }

    generated = generate_from_btoks_cache(
        encode_out=encode_out,
        backbone_model=backbone,
        prompt_batch=prompt_batch,
        eos_token_id=eos_token_id,
        max_new_tokens=2,
    )

    assert generated.tolist() == [[eos_token_id]]
    assert backbone.calls[0]["position_ids"].shape == (3, 1, 2)
    assert backbone.calls[0]["position_ids"].tolist() == [[[3, 4]], [[3, 4]], [[3, 4]]]


def test_generate_from_btoks_cache_rejects_missing_btoks_tokens():
    backbone = FakeBackboneModel(next_token_ids=[1], eos_token_id=2)
    encode_out = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "btoks_token_mask": torch.tensor([[0, 0]], dtype=torch.bool),
        "past_key_values": _make_cache(seq_len=2),
    }
    prompt_batch = {
        "input_ids": torch.tensor([[20]], dtype=torch.long),
        "attention_mask": torch.tensor([[1]], dtype=torch.long),
    }

    try:
        generate_from_btoks_cache(
            encode_out=encode_out,
            backbone_model=backbone,
            prompt_batch=prompt_batch,
            eos_token_id=2,
            max_new_tokens=1,
        )
    except ValueError as err:
        assert "No btoks-token KV entries" in str(err)
    else:
        raise AssertionError("Expected ValueError when encode_out has no btoks tokens")


def test_generate_from_btoks_cache_stops_on_additional_stop_token():
    backbone = FakeBackboneModel(next_token_ids=[9, 7], eos_token_id=2)
    encode_out = {
        "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "btoks_token_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
        "past_key_values": _make_cache(seq_len=3),
    }
    prompt_batch = {
        "input_ids": torch.tensor([[20, 21, 22]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    generated = generate_from_btoks_cache(
        encode_out=encode_out,
        backbone_model=backbone,
        prompt_batch=prompt_batch,
        eos_token_id=2,
        stop_token_ids=[9],
        max_new_tokens=4,
    )

    assert generated.tolist() == [[9]]
