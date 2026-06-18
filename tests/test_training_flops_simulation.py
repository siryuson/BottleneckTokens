from __future__ import annotations

from vlm2emb.analysis.training_flops import (
    FlopsConfig,
    FlopsEstimator,
    StepFlops,
    _estimate_btoks_step,
    _estimate_vlm2vec_step,
    aggregate_windows,
)


def _estimator() -> FlopsEstimator:
    return FlopsEstimator(
        FlopsConfig(
            num_hidden_layers=2,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            vision_depth=1,
            vision_embed_dim=8,
            vision_num_heads=2,
        ),
        train_forward_backward_multiplier=3.0,
    )


def _sample(length: int, vision_tokens: list[int] | None = None) -> dict:
    return {
        "input_length": length,
        "vision_segment_tokens": list(vision_tokens or []),
    }


def test_grad_cache_estimate_splits_phase_a_and_phase_b() -> None:
    step = _estimate_vlm2vec_step(
        estimator=_estimator(),
        config_id="cfg",
        trainer_type="vlm2vec_trainer",
        step=1,
        query_samples=[_sample(4), _sample(8)],
        positive_samples=[_sample(5), _sample(7)],
        negative_samples=[None, None],
        gc_chunk_size=1,
        world_size=2,
    )

    assert step.grad_cache_phase_a_flops > 0
    assert step.grad_cache_phase_b_flops == step.grad_cache_phase_a_flops * 3
    assert step.global_flops == step.per_device_flops * 2
    assert step.btoks_generation_flops == 0


def test_btoks_generation_respects_duration() -> None:
    config = {
        "model": {"modules": {"injector": {"num_tokens": 4}}},
        "train": {
            "args": {
                "generation_loss_weight": 1.0,
                "generation_loss_duration": 1,
                "generation_kv_mode": "compressed",
                "generation_prefix_text": "<p>",
                "generation_suffix_text": "</p>",
            }
        },
    }

    class Tokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False):
            return {"input_ids": [1, 2] if text else []}

    class Wrapper:
        tokenizer = Tokenizer()

    enabled = _estimate_btoks_step(
        estimator=_estimator(),
        config=config,
        config_id="cfg",
        trainer_type="btoks_trainer",
        step=1,
        query_samples=[_sample(4)],
        positive_samples=[_sample(5)],
        negative_samples=[None],
        metadata={"ntp_side": ["positive"]},
        wrapper=Wrapper(),
        gc_chunk_size=1,
        world_size=1,
    )
    disabled = _estimate_btoks_step(
        estimator=_estimator(),
        config=config,
        config_id="cfg",
        trainer_type="btoks_trainer",
        step=2,
        query_samples=[_sample(4)],
        positive_samples=[_sample(5)],
        negative_samples=[None],
        metadata={"ntp_side": ["positive"]},
        wrapper=Wrapper(),
        gc_chunk_size=1,
        world_size=1,
    )

    assert enabled.btoks_generation_flops > 0
    assert enabled.generation_selected == 1
    assert disabled.btoks_generation_flops == 0
    assert disabled.generation_selected == 0


def test_window_aggregation_keeps_partial_window() -> None:
    steps = [
        StepFlops(
            config_id="cfg",
            trainer_type="trainer",
            step=idx + 1,
            per_device_flops=1_000_000_000.0,
            global_flops=2_000_000_000.0,
            grad_cache_phase_a_flops=0,
            grad_cache_phase_b_flops=0,
            btoks_generation_flops=0,
            query_padded_len_max=1,
            positive_padded_len_max=1,
            negative_padded_len_max=0,
        )
        for idx in range(3)
    ]

    windows = aggregate_windows(steps, 2)

    assert len(windows) == 2
    assert windows[0].steps == 2
    assert windows[0].global_gflops == 4
    assert windows[1].steps == 1
    assert windows[1].global_gflops == 2
