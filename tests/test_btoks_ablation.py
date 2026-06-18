from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from vlm2emb.auto import AutoTrainingArgs
from vlm2emb.config import apply_overrides, load_config
from vlm2emb.modules.btoks import (
    BToksAttentionPooling,
    BToksPooling,
    BToksTokenInjector,
    extract_last_token_kv,
)
from vlm2emb.training.trainers.btoks_trainer import BToksTrainingArgs, _btoks_forward_fn


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_cache(values: torch.Tensor) -> DynamicCache:
    """Build a one-layer DynamicCache from a (B, L) tensor."""
    cache = DynamicCache()
    key = values.to(dtype=torch.float32).view(values.shape[0], 1, values.shape[1], 1)
    value = key + 100
    cache.update(key, value, layer_idx=0)
    return cache


def test_btoks_token_injector_zero_tokens_skips_setup_and_keeps_inputs() -> None:
    class FailIfTouched:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected access to {name}")

    injector = BToksTokenInjector(num_tokens=0)
    injector.on_processor(model=FailIfTouched(), processor=FailIfTouched())

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    output = injector(input_ids=input_ids, attention_mask=attention_mask)

    assert output["input_ids"] is input_ids
    assert output["attention_mask"] is attention_mask
    assert output["btoks_token_mask"].shape == input_ids.shape
    assert not output["btoks_token_mask"].any()
    assert injector.get_token_ids() is None
    assert injector.get_trainable_token_indices() is None


def test_btoks_pooling_zero_tokens_uses_last_valid_token() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.long,
    )
    btoks_mask = torch.zeros(2, 4, dtype=torch.bool)

    output = BToksPooling()(hidden, btoks_token_mask=btoks_mask, attention_mask=attention_mask)

    expected = torch.stack([hidden[0, 1], hidden[1, 3]])
    assert torch.equal(output["embeddings"], expected)


def test_btoks_pooling_zero_tokens_without_attention_mask_uses_sequence_tail() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    output = BToksPooling()(hidden, btoks_token_mask=torch.zeros(2, 4, dtype=torch.bool))

    assert torch.equal(output["embeddings"], hidden[:, -1, :])


def test_btoks_attention_pooling_zero_tokens_uses_last_valid_token() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)

    output = BToksAttentionPooling(hidden_dim=3)(
        hidden,
        btoks_token_mask=torch.zeros(2, 4, dtype=torch.bool),
        attention_mask=attention_mask,
    )

    assert torch.equal(output["embeddings"], torch.stack([hidden[0, 1], hidden[1, 2]]))


def test_extract_last_token_kv_supports_per_sample_positions() -> None:
    cache = _make_cache(torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]]))
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)

    selected = extract_last_token_kv(
        past_key_values=cache,
        attention_mask=attention_mask,
        seq_len=4,
        batch_size=2,
        device=attention_mask.device,
    )
    key, value = next(iter(selected))

    assert key.shape == (2, 1, 1, 1)
    assert key[:, 0, 0, 0].tolist() == [1.0, 12.0]
    assert value[:, 0, 0, 0].tolist() == [101.0, 112.0]


class _RecordingBackboneModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(loss=torch.tensor(0.5))


class _RecordingBackbone:
    def __init__(self) -> None:
        self.model = _RecordingBackboneModel()

    def build_continuation_position_ids(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor,
        cache_position: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return start_positions.view(1, -1, 1) + torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).view(1, 1, -1)


class _UnwrappedForGeneration:
    def __init__(
        self,
        *,
        btoks_token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        self.backbone = _RecordingBackbone()
        self._modules_dict = {"backbone": self.backbone}
        self.btoks_token_mask = btoks_token_mask
        self.attention_mask = attention_mask

    def encode(self, **kwargs) -> dict:
        cache_values = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [10, 11, 12, 13, 14],
            ],
            dtype=torch.float32,
        )
        return {
            "embeddings": torch.ones(2, 4),
            "attention_mask": self.attention_mask,
            "past_key_values": _make_cache(cache_values),
            "btoks_token_mask": self.btoks_token_mask,
        }


def _run_generation_forward(
    unwrapped: _UnwrappedForGeneration,
    *,
    generation_kv_mode: str,
) -> dict:
    opposite_ids = torch.tensor([[21, 22], [23, 24]], dtype=torch.long)
    return _btoks_forward_fn(
        unwrapped,  # type: ignore[arg-type]
        selected_indices=[0, 1],
        opposite_batch={
            "input_ids": opposite_ids,
            "attention_mask": torch.ones_like(opposite_ids),
        },
        labels=opposite_ids.clone(),
        generation_kv_mode=generation_kv_mode,
        input_ids=torch.ones(2, 5, dtype=torch.long),
        attention_mask=unwrapped.attention_mask,
    )


def test_btoks_forward_default_mask_uses_btoks_token_kv() -> None:
    unwrapped = _UnwrappedForGeneration(
        btoks_token_mask=torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, True, True],
            ]
        ),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    output = _run_generation_forward(unwrapped, generation_kv_mode="compressed")
    call = unwrapped.backbone.model.calls[0]
    key, _ = next(iter(call["past_key_values"]))

    assert output["kv_cache_mode"] == "btoks_token"
    assert key.shape[2] == 2
    assert key[:, 0, :, 0].tolist() == [[3.0, 4.0], [13.0, 14.0]]


def test_btoks_forward_can_use_full_kv_cache() -> None:
    attention_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    unwrapped = _UnwrappedForGeneration(
        btoks_token_mask=torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, True, True],
            ]
        ),
        attention_mask=attention_mask,
    )

    output = _run_generation_forward(unwrapped, generation_kv_mode="full")
    call = unwrapped.backbone.model.calls[0]
    key, _ = next(iter(call["past_key_values"]))

    assert output["kv_cache_mode"] == "full"
    assert key.shape[2] == 5
    assert key[:, 0, :, 0].tolist() == [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
    ]
    assert call["attention_mask"].tolist() == [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1],
    ]


def test_btoks_forward_zero_tokens_with_mask_uses_last_token_kv() -> None:
    attention_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    unwrapped = _UnwrappedForGeneration(
        btoks_token_mask=torch.zeros(2, 5, dtype=torch.bool),
        attention_mask=attention_mask,
    )

    output = _run_generation_forward(unwrapped, generation_kv_mode="compressed")
    call = unwrapped.backbone.model.calls[0]
    key, _ = next(iter(call["past_key_values"]))

    assert output["kv_cache_mode"] == "last_token"
    assert key.shape == (2, 1, 1, 1)
    assert key[:, 0, 0, 0].tolist() == [1.0, 12.0]


def test_btoks_training_args_generation_kv_mode_default_and_override() -> None:
    default_args = AutoTrainingArgs.from_config(
        {
            "type": "btoks",
            "output_dir": "/tmp/btoks-default",
            "report_to": "none",
        }
    )
    override_args = AutoTrainingArgs.from_config(
        {
            "type": "btoks",
            "output_dir": "/tmp/btoks-full-kv",
            "report_to": "none",
            "generation_kv_mode": "full",
        }
    )

    assert isinstance(default_args, BToksTrainingArgs)
    assert default_args.generation_kv_mode == "compressed"
    assert isinstance(override_args, BToksTrainingArgs)
    assert override_args.generation_kv_mode == "full"


def test_btoks_training_args_rejects_old_use_btoks_kv_mask_field() -> None:
    with pytest.raises(TypeError, match="use_btoks_kv_mask"):
        AutoTrainingArgs.from_config(
            {
                "type": "btoks",
                "output_dir": "/tmp/btoks-old-field",
                "report_to": "none",
                "use_btoks_kv_mask": False,
            }
        )


def test_btoks_training_args_rejects_invalid_generation_kv_mode() -> None:
    with pytest.raises(ValueError, match="generation_kv_mode"):
        AutoTrainingArgs.from_config(
            {
                "type": "btoks",
                "output_dir": "/tmp/btoks-invalid-mode",
                "report_to": "none",
                "generation_kv_mode": "masked",
            }
        )


def test_btoks_default_preset_keeps_default_ablation_controls() -> None:
    default = load_config(REPO_ROOT / "configs/presets/btoks_qwen2vl_2b_v1.yaml")

    assert default.model.modules.injector.num_tokens == 4
    assert default.train.args.generation_kv_mode == "compressed"


def test_btoks_ablation_controls_can_be_expressed_as_cli_overrides() -> None:
    config = load_config(REPO_ROOT / "configs/presets/btoks_qwen2vl_2b_v1.yaml")

    no_bottleneck = apply_overrides(
        config,
        [
            "model.modules.injector.num_tokens=0",
        ],
    )
    assert no_bottleneck.model.modules.injector.num_tokens == 0
    assert no_bottleneck.train.args.generation_kv_mode == "compressed"

    full_kv = apply_overrides(
        config,
        [
            "train.args.generation_kv_mode=full",
        ],
    )
    assert full_kv.model.modules.injector.num_tokens == 4
    assert full_kv.train.args.generation_kv_mode == "full"

    combined = apply_overrides(
        config,
        [
            "model.modules.injector.num_tokens=0",
            "train.args.generation_kv_mode=full",
        ],
    )
    assert combined.model.modules.injector.num_tokens == 0
    assert combined.train.args.generation_kv_mode == "full"
