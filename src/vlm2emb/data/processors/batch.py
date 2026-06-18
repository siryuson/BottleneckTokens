"""Batch processing utilities for ProcessorWrapper outputs.

ProcessorWrapper 

This module provides batch_processor_outputs() to concatenate per-sample
ProcessorWrapper outputs into batched format for model forward.

 batch_processor_outputs() ProcessorWrapper 
 forward

Note: Normally transformers processors handle batching internally, but
ProcessorWrapper returns per-sample results for flexible batching (e.g.,
GradCache chunks). This function manually reimplements the batching process
and MUST align with the native processor's batching behavior (padding
strategy, attention_mask generation, pixel_values concatenation, etc.).

 transformers processor  ProcessorWrapper

 processor 
"""

from __future__ import annotations

from typing import Any

import torch


def batch_processor_outputs(
    samples: list[dict[str, torch.Tensor | None]],
    wrapper: Any,
) -> dict[str, torch.Tensor | None]:
    """Concatenate ProcessorWrapper outputs into batched format for model forward.

     ProcessorWrapper  forward

    Args:
        samples: List of sample dicts from ProcessorWrapper.__call__().
                 ProcessorWrapper.__call__() 
        wrapper: ProcessorWrapper instance. Uses wrapper.tokenizer for padding.
                 ProcessorWrapper  wrapper.tokenizer 

    Returns:
        Batched dict with:
        - input_ids: Padded token IDs [batch_size, seq_len]
        - attention_mask: Attention mask [batch_size, seq_len]
        - pixel_values: Concatenated image pixels or None
        - image_grid_thw: Concatenated image grid info or None
        - pixel_values_videos: Concatenated video pixels or None
        - video_grid_thw: Concatenated video grid info or None

    Example:
        >>> wrapper = Qwen2VLProcessorWrapper(processor)
        >>> samples = wrapper(texts, images)  # List[Dict]
        >>> batched = batch_processor_outputs(samples, wrapper)
        >>> output = model(**batched)
    """
    if not samples:
        raise ValueError("samples list is empty")

    tokenizer = wrapper.tokenizer

    # Extract and prepare input_ids for padding.
    input_ids_list: list[list[int]] = []
    for sample in samples:
        input_ids = sample.get("input_ids")
        if input_ids is None:
            raise ValueError("Each sample must include input_ids")
        ids = input_ids.squeeze().tolist()
        input_ids_list.append(ids if isinstance(ids, list) else [ids])

    # Pad input_ids and generate attention_mask.
    padded = tokenizer.pad(
        {"input_ids": input_ids_list},
        return_tensors="pt",
    )
    seq_len = padded["input_ids"].shape[1]
    padding_side = getattr(tokenizer, "padding_side", "right")

    # Concatenate visual tensors (filter out None values)
    def cat_if_exists(key: str) -> torch.Tensor | None:
        tensors: list[torch.Tensor] = []
        for sample in samples:
            value = sample.get(key)
            if value is not None:
                tensors.append(value)
        return torch.cat(tensors, dim=0) if tensors else None

    def pad_token_aligned_field(key: str, pad_value: int = 0) -> torch.Tensor | None:
        if not any(sample.get(key) is not None for sample in samples):
            return None

        padded_tensors: list[torch.Tensor] = []
        for sample in samples:
            value = sample.get(key)
            if value is None:
                input_ids = sample.get("input_ids")
                if input_ids is None:
                    raise ValueError(f"Each sample must include input_ids when padding {key}")
                value = torch.zeros_like(input_ids, dtype=torch.long)
            value = value.long()
            if value.ndim == 2 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.ndim != 1:
                raise ValueError(f"{key} must be 1D or [1, L], got shape {tuple(value.shape)}")

            pad_len = seq_len - value.shape[0]
            if pad_len < 0:
                raise ValueError(f"{key} length {value.shape[0]} exceeds padded input length {seq_len}")
            if pad_len:
                pad_tensor = torch.full(
                    (pad_len,),
                    pad_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                if padding_side == "left":
                    value = torch.cat([pad_tensor, value], dim=0)
                else:
                    value = torch.cat([value, pad_tensor], dim=0)
            padded_tensors.append(value)

        return torch.stack(padded_tensors, dim=0)

    return {
        "input_ids": padded["input_ids"].long(),
        "attention_mask": padded["attention_mask"].long(),
        "generation_loss_mask": pad_token_aligned_field("generation_loss_mask"),
        "mm_token_type_ids": pad_token_aligned_field("mm_token_type_ids"),
        "pixel_values": cat_if_exists("pixel_values"),
        "image_grid_thw": cat_if_exists("image_grid_thw"),
        "pixel_values_videos": cat_if_exists("pixel_values_videos"),
        "video_grid_thw": cat_if_exists("video_grid_thw"),
        "second_per_grid_ts": cat_if_exists("second_per_grid_ts"),
    }
