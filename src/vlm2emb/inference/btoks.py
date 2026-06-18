"""BToks bottleneck-cache generation helpers.

This module provides utilities for decoding text from the bottleneck-token KV
cache produced by a BToks encoder pass.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from transformers import DynamicCache

from vlm2emb.modules.btoks import extract_btoks_kv


def _select_batch_from_cache(cache: DynamicCache, batch_idx: Tensor) -> DynamicCache:
    """Subset a DynamicCache along the batch dimension."""
    selected = DynamicCache()
    for layer_idx, (key_states, value_states) in enumerate(cache):
        selected.update(
            key_states.index_select(0, batch_idx),
            value_states.index_select(0, batch_idx),
            layer_idx=layer_idx,
        )
    return selected


def _subset_prompt_batch(
    prompt_batch: dict[str, Tensor | None],
    batch_idx: Tensor,
) -> dict[str, Tensor]:
    """Select prompt tensors for the requested batch indices."""
    subset: dict[str, Tensor] = {}
    for key, value in prompt_batch.items():
        if value is None:
            continue
        if not torch.is_tensor(value):
            continue
        subset[key] = value.index_select(0, batch_idx)
    return subset


def _compute_start_positions(
    encode_out: dict[str, Any],
    batch_idx: Tensor,
) -> Tensor:
    """Compute RoPE offsets for continuation tokens."""
    encode_mask = encode_out.get("attention_mask")
    if encode_mask is not None:
        return encode_mask.sum(dim=1)[batch_idx].to(dtype=torch.long)

    encode_input_ids = encode_out.get("input_ids")
    if encode_input_ids is None:
        raise ValueError("encode_out must include attention_mask or input_ids")

    seq_len = encode_input_ids.shape[1]
    return torch.full(
        (batch_idx.shape[0],),
        seq_len,
        device=encode_input_ids.device,
        dtype=torch.long,
    )


def _build_position_ids(
    backbone_model: Any,
    prompt_batch: dict[str, Tensor],
    prompt_attention_mask: Tensor,
    start_positions: Tensor,
) -> Tensor:
    """Build continuation position ids for prompt prefill."""
    prompt_input_ids = prompt_batch["input_ids"]
    rope_model = backbone_model
    if not hasattr(rope_model, "get_rope_index") and hasattr(backbone_model, "model"):
        rope_model = backbone_model.model

    if hasattr(rope_model, "get_rope_index"):
        position_ids, _ = rope_model.get_rope_index(
            input_ids=prompt_input_ids,
            image_grid_thw=prompt_batch.get("image_grid_thw"),
            video_grid_thw=prompt_batch.get("video_grid_thw"),
            attention_mask=prompt_attention_mask,
        )
        return position_ids + start_positions.view(1, -1, 1)

    offsets = torch.arange(
        prompt_input_ids.shape[1],
        device=prompt_input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    return offsets + start_positions.unsqueeze(1)


def _sample_next_token(
    logits: Tensor,
    *,
    do_sample: bool,
    temperature: float,
) -> Tensor:
    """Select the next token from final-step logits."""
    if not do_sample or temperature <= 0:
        return logits.argmax(dim=-1)

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def generate_from_btoks_cache(
    *,
    encode_out: dict[str, Any],
    backbone_model: Any,
    prompt_batch: dict[str, Tensor | None],
    eos_token_id: int,
    max_new_tokens: int,
    selected_indices: list[int] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    stop_token_ids: list[int] | None = None,
) -> Tensor:
    """Generate tokens from bottleneck-token KV cache.

    Args:
        encode_out: Output from ``model.encode(..., return_cache=True)``.
        backbone_model: Underlying causal LM model used for continuation decode.
        prompt_batch: Text prompt batch used to start continuation.
        eos_token_id: End-of-sequence token id.
        max_new_tokens: Maximum number of generated tokens.
        selected_indices: Optional subset of encode batch indices to decode.
        do_sample: Whether to sample from the next-token distribution.
        temperature: Sampling temperature. Greedy decoding is used when
            ``do_sample`` is false or temperature <= 0.
        stop_token_ids: Optional additional token ids that should terminate
            generation, e.g. ``<|im_end|>`` for chat-style assistant spans.

    Returns:
        Generated token ids with shape ``(B, T_generated)``.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    extra_stop_ids = {
        int(token_id)
        for token_id in (stop_token_ids or [])
        if token_id is not None and int(token_id) >= 0
    }

    past_key_values = encode_out.get("past_key_values")
    btoks_token_mask = encode_out.get("btoks_token_mask")
    if past_key_values is None or btoks_token_mask is None:
        raise ValueError("encode_out must include past_key_values and btoks_token_mask")

    prompt_input_ids = prompt_batch.get("input_ids")
    if prompt_input_ids is None:
        raise ValueError("prompt_batch must include input_ids")
    if prompt_input_ids.shape[1] < 1:
        raise ValueError("prompt_batch.input_ids must contain at least one token")

    prompt_attention_mask = prompt_batch.get("attention_mask")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_input_ids)

    btoks_kv = extract_btoks_kv(
        past_key_values=past_key_values,
        btoks_token_mask=btoks_token_mask,
    )
    if btoks_kv is None:
        raise ValueError("No btoks-token KV entries found in encode_out")

    batch_size = prompt_input_ids.shape[0]
    device = prompt_input_ids.device
    if selected_indices is None:
        batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
    else:
        batch_idx = torch.tensor(selected_indices, device=device, dtype=torch.long)
    if batch_idx.numel() < 1:
        raise ValueError("selected_indices must not be empty")

    selected_kv = _select_batch_from_cache(btoks_kv, batch_idx)
    prompt_subset = _subset_prompt_batch(prompt_batch, batch_idx)
    prompt_input_ids = prompt_subset["input_ids"]
    prompt_attention_mask = prompt_subset.get("attention_mask", prompt_attention_mask.index_select(0, batch_idx))

    start_positions = _compute_start_positions(encode_out, batch_idx).to(device=device)
    position_ids = _build_position_ids(
        backbone_model=backbone_model,
        prompt_batch=prompt_subset,
        prompt_attention_mask=prompt_attention_mask,
        start_positions=start_positions,
    )

    btoks_cache_len = selected_kv.get_seq_length()
    prefix_mask = torch.ones(
        prompt_input_ids.shape[0],
        btoks_cache_len,
        dtype=prompt_attention_mask.dtype,
        device=device,
    )
    full_attention_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)
    cache_position = torch.arange(
        btoks_cache_len,
        btoks_cache_len + prompt_input_ids.shape[1],
        device=device,
        dtype=torch.long,
    )

    prompt_inputs = {
        key: value
        for key, value in prompt_subset.items()
        if key != "attention_mask"
    }
    outputs = backbone_model(
        **prompt_inputs,
        attention_mask=full_attention_mask,
        past_key_values=selected_kv,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    generated_tokens: list[Tensor] = []
    finished = torch.zeros(prompt_input_ids.shape[0], dtype=torch.bool, device=device)
    current_cache = outputs.past_key_values
    current_attention_mask = full_attention_mask
    next_position_ids = position_ids[..., -1:]

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :]
        next_tokens = _sample_next_token(
            logits,
            do_sample=do_sample,
            temperature=temperature,
        )
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, eos_token_id),
            next_tokens,
        )
        generated_tokens.append(next_tokens)
        finished = finished | next_tokens.eq(eos_token_id)
        for stop_token_id in extra_stop_ids:
            finished = finished | next_tokens.eq(stop_token_id)
        if torch.all(finished):
            break

        current_cache_len = current_cache.get_seq_length()
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    prompt_input_ids.shape[0],
                    1,
                    dtype=current_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        next_position_ids = next_position_ids + 1
        outputs = backbone_model(
            input_ids=next_tokens.unsqueeze(1),
            attention_mask=current_attention_mask,
            past_key_values=current_cache,
            position_ids=next_position_ids,
            cache_position=torch.tensor([current_cache_len], device=device, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
        current_cache = outputs.past_key_values

    return torch.stack(generated_tokens, dim=1)
