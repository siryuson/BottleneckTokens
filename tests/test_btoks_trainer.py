from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding, DynamicCache

from vlm2emb.modules.backbone import BackboneBase, Qwen3VLBackbone
from vlm2emb.training.grad_cache import ChunkContext, grad_cache_accumulate
from vlm2emb.training.trainers.btoks_trainer import BToksTrainer, _btoks_forward_fn


def _make_trainer(
    *,
    include_generation_suffix_in_loss: bool = True,
    visual_token_ids: set[int] | None = None,
) -> BToksTrainer:
    trainer = object.__new__(BToksTrainer)
    trainer.generation_prefix_ids = [101, 102, 103]
    trainer.generation_suffix_ids = [104]
    trainer.include_generation_suffix_in_loss = include_generation_suffix_in_loss
    trainer._generation_visual_token_ids = visual_token_ids or set()
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    return trainer


def test_decorate_generation_sample_wraps_protocol_and_masks_visual_tokens():
    trainer = _make_trainer(visual_token_ids={151652, 151653, 151655, 151656})
    sample = {
        "input_ids": torch.tensor([[151652, 151655, 151653, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.long),
    }

    decorated = trainer._decorate_generation_sample(sample)

    assert decorated["input_ids"].tolist() == [[101, 102, 103, 151652, 151655, 151653, 11, 12, 104]]
    assert decorated["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
    assert decorated["generation_loss_mask"].tolist() == [[0, 0, 0, 0, 0, 0, 1, 1, 1]]
    assert decorated["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 1, 0, 0, 0, 0]]


def test_decorate_generation_sample_can_exclude_suffix_from_loss():
    trainer = _make_trainer(include_generation_suffix_in_loss=False)
    sample = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }

    decorated = trainer._decorate_generation_sample(sample)

    assert decorated["generation_loss_mask"].tolist() == [[0, 0, 0, 1, 1, 0]]


def test_build_generation_labels_prefers_generation_loss_mask():
    trainer = _make_trainer()
    opposite_batch = {
        "input_ids": torch.tensor([[101, 102, 103, 11, 12, 104]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        "generation_loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long),
    }

    labels = trainer._build_generation_labels(opposite_batch, model=SimpleNamespace())

    assert labels.tolist() == [[-100, -100, -100, 11, 12, 104]]


def test_encode_generation_text_accepts_transformers_batch_encoding():
    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> BatchEncoding:
            assert text == "<|im_start|>"
            assert add_special_tokens is False
            return BatchEncoding({"input_ids": [151644]})

    assert BToksTrainer._encode_generation_text(
        Tokenizer(),
        "<|im_start|>",
        field_name="generation_prefix_text",
    ) == [151644]


def test_encode_generation_text_flattens_single_batch_tensor():
    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, torch.Tensor]:
            assert text == "<|im_end|>"
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[151645]], dtype=torch.long)}

    assert BToksTrainer._encode_generation_text(
        Tokenizer(),
        "<|im_end|>",
        field_name="generation_suffix_text",
    ) == [151645]


def test_pack_inputs_treats_ntp_side_as_generation_target_side():
    trainer = _make_trainer()
    inputs = {
        "query": [{"qid": 1}, {"qid": 2}, {"qid": 3}, {"qid": 4}, {"qid": 5}],
        "positive": [{"pid": 1}, {"pid": 2}, {"pid": 3}, {"pid": 4}, {"pid": 5}],
        "metadata": {"ntp_side": ["query", "positive", "both", "none", "target"]},
    }

    query_group, positive_group = trainer._pack_inputs(inputs)

    assert [sample["should_generate"] for sample in query_group] == [
        False,
        True,
        True,
        False,
        True,
    ]
    assert [sample["should_generate"] for sample in positive_group] == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_pack_inputs_rejects_mismatched_query_positive_lengths():
    trainer = _make_trainer()

    with pytest.raises(ValueError, match="same number of samples"):
        trainer._pack_inputs(
            {
                "query": [{"qid": 1}, {"qid": 2}],
                "positive": [{"pid": 1}],
                "metadata": {"ntp_side": ["query", "positive"]},
            }
        )


def test_packed_self_token_length_uses_self_side_attention_mask():
    packed = {
        "self": {
            "input_ids": torch.ones((1, 6), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.long),
        },
        "opposite": {"input_ids": torch.ones((1, 2), dtype=torch.long)},
    }

    assert BToksTrainer._packed_self_token_length(packed) == 3


def test_btoks_grad_cache_sorting_preserves_generation_pair_alignment():
    trainer = _make_trainer()

    def sample(prefix: str, pair_id: int, length: int) -> dict:
        return {
            "name": f"{prefix}{pair_id}",
            "pair_id": pair_id,
            "input_ids": torch.full((1, length), pair_id, dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    packed_inputs = trainer._pack_inputs(
        {
            "query": [
                sample("q", 0, 5),
                sample("q", 1, 2),
                sample("q", 2, 7),
                sample("q", 3, 2),
            ],
            "positive": [
                sample("p", 0, 10),
                sample("p", 1, 3),
                sample("p", 2, 4),
                sample("p", 3, 9),
            ],
            "metadata": {"ntp_side": ["none", "positive", "both", "query"]},
        }
    )
    records: list[dict] = []

    def encode(chunk: list[dict]) -> torch.Tensor:
        values = [float(sample["self"]["pair_id"]) for sample in chunk]
        return torch.tensor(values, dtype=torch.float32).view(-1, 1)

    def forward_backward(
        chunk: list[dict],
        cache: torch.Tensor,
        ctx: ChunkContext,
    ) -> None:
        selected_indices = [
            idx for idx, packed in enumerate(chunk)
            if packed.get("should_generate", False)
        ]
        records.append(
            {
                "group_idx": ctx.group_idx,
                "self": [packed["self"]["name"] for packed in chunk],
                "opposite": [packed["opposite"]["name"] for packed in chunk],
                "selected": selected_indices,
                "selected_pairs": [
                    chunk[idx]["self"]["pair_id"] for idx in selected_indices
                ],
                "selected_opposites": [
                    chunk[idx]["opposite"]["name"] for idx in selected_indices
                ],
            }
        )

    grad_cache_accumulate(
        inputs=packed_inputs,
        encode_fns=[encode, encode],
        loss_fn=lambda *reps: sum(rep.sum() for rep in reps),
        chunk_sizes=[2, 2],
        forward_backward_fns=[forward_backward, forward_backward],
        aligned_sort_key_fn=BToksTrainer._packed_self_token_length,
        aligned_sort_group_idx=0,
    )

    assert records == [
        {
            "group_idx": 0,
            "self": ["q1", "q3"],
            "opposite": ["p1", "p3"],
            "selected": [0],
            "selected_pairs": [1],
            "selected_opposites": ["p1"],
        },
        {
            "group_idx": 0,
            "self": ["q0", "q2"],
            "opposite": ["p0", "p2"],
            "selected": [1],
            "selected_pairs": [2],
            "selected_opposites": ["p2"],
        },
        {
            "group_idx": 1,
            "self": ["p1", "p3"],
            "opposite": ["q1", "q3"],
            "selected": [1],
            "selected_pairs": [3],
            "selected_opposites": ["q3"],
        },
        {
            "group_idx": 1,
            "self": ["p0", "p2"],
            "opposite": ["q0", "q2"],
            "selected": [1],
            "selected_pairs": [2],
            "selected_opposites": ["q2"],
        },
    ]


def test_btoks_forward_uses_inner_rope_index_for_qwen3_generation() -> None:
    class InnerModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, torch.Tensor | None]] = []

        def get_rope_index(
            self,
            *,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor | None = None,
            video_grid_thw: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls.append(
                {
                    "input_ids": input_ids,
                    "image_grid_thw": image_grid_thw,
                    "video_grid_thw": video_grid_thw,
                    "attention_mask": attention_mask,
                }
            )
            base = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1)
            return base.expand(3, input_ids.shape[0], -1), torch.zeros(input_ids.shape[0], 1)

    class TopModel:
        def __init__(self) -> None:
            self.model = InnerModel()
            self.calls: list[dict[str, torch.Tensor | None]] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(loss=torch.tensor(0.25))

    class FakeBackbone(BackboneBase):
        def __init__(self) -> None:
            super().__init__()
            self.model = TopModel()

    class Unwrapped:
        def __init__(self) -> None:
            self.backbone = FakeBackbone()
            self._modules_dict = {"backbone": self.backbone}

        def encode(self, **kwargs) -> dict[str, torch.Tensor | DynamicCache]:
            cache = DynamicCache()
            key = torch.arange(20, dtype=torch.float32).view(2, 1, 5, 2)
            value = key + 100
            cache.update(key, value, layer_idx=0)
            return {
                "embeddings": torch.ones(2, 4),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                    ],
                    dtype=torch.long,
                ),
                "past_key_values": cache,
                "btoks_token_mask": torch.tensor(
                    [
                        [False, False, False, True, True],
                        [False, False, False, True, True],
                    ]
                ),
            }

    unwrapped = Unwrapped()
    opposite_ids = torch.tensor([[151652, 151655, 151653]], dtype=torch.long)

    outputs = _btoks_forward_fn(
        unwrapped,  # type: ignore[arg-type]
        selected_indices=[1],
        opposite_batch={
            "input_ids": opposite_ids,
            "attention_mask": torch.ones_like(opposite_ids),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
        labels=opposite_ids.clone(),
        input_ids=torch.ones(2, 3, dtype=torch.long),
        attention_mask=torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )

    inner = unwrapped.backbone.model.model
    assert len(inner.calls) == 1
    assert torch.equal(inner.calls[0]["input_ids"], opposite_ids)
    forward_call = unwrapped.backbone.model.calls[0]
    assert forward_call["position_ids"].shape == (3, 1, 3)
    assert forward_call["position_ids"].tolist() == [
        [[4, 5, 6]],
        [[4, 5, 6]],
        [[4, 5, 6]],
    ]
    assert outputs["gen_loss"].item() == pytest.approx(0.25)


def test_qwen3_continuation_position_ids_prepend_cache_positions() -> None:
    class RopeOwner:
        def get_rope_index(
            self,
            *,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor | None = None,
            video_grid_thw: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            base = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1)
            return base.expand(3, input_ids.shape[0], -1), torch.zeros(input_ids.shape[0], 1)

    dummy_model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=4),
            vision_config=SimpleNamespace(hidden_size=8),
        ),
        model=RopeOwner(),
    )
    backbone = Qwen3VLBackbone(model=dummy_model)  # type: ignore[arg-type]

    position_ids = backbone.build_continuation_position_ids(
        input_ids=torch.ones(2, 3, dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        start_positions=torch.tensor([5, 4]),
        cache_position=torch.tensor([2, 3, 4]),
    )

    assert position_ids.shape == (4, 2, 3)
    assert position_ids[0].tolist() == [
        [2, 3, 4],
        [2, 3, 4],
    ]
    assert position_ids[1:].tolist() == [
        [[5, 6, 7], [4, 5, 6]],
        [[5, 6, 7], [4, 5, 6]],
        [[5, 6, 7], [4, 5, 6]],
    ]
