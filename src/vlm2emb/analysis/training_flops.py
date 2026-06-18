"""Offline training FLOPs simulation.

The simulator reuses the production dataset/collator/processor path to collect
token and vision grid shapes, then estimates Qwen2-VL training FLOPs from those
shapes without loading model weights or running training.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from io import BytesIO
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vlm2emb.auto import AutoCollator, AutoDataset, AutoProcessorWrapper
from vlm2emb.config import ConfigLoader, apply_overrides, to_native_config
from vlm2emb.data.collators.training_collator import (
    _build_train_field_payload,
    render_instruction_text,
)
from vlm2emb.data.processors.media import extract_multimodal_payload
from vlm2emb.training.samplers import BatchInterleaveSampler

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = (
    "configs/presets/vlm2vec_qwen2vl_2b.yaml",
    "configs/presets/btoks_qwen2vl_2b_v1.yaml",
)
FLOPS_PER_GIGA = 1_000_000_000.0
FLOPS_PER_TERA = 1_000_000_000_000.0


@dataclass(frozen=True)
class SimulationOptions:
    """Runtime options for one offline FLOPs simulation."""

    max_steps: int = 5000
    window_steps: int = 500
    batch_size: int | None = None
    world_size: int = 1
    rank: int = 0
    drop_last: bool | None = None
    train_forward_backward_multiplier: float = 3.0
    progress_every: int = 100
    overrides: tuple[str, ...] = ()
    exact_processor: bool = False
    cache_source_shapes: bool = False


@dataclass(frozen=True)
class FlopsConfig:
    """Model dimensions used by the static FLOPs estimator."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vision_depth: int = 0
    vision_embed_dim: int = 0
    vision_num_heads: int = 0
    vision_mlp_ratio: float = 4.0
    vision_spatial_merge_size: int = 2
    multiply_add_flops: int = 2

    @property
    def head_dim(self) -> int:
        """Return attention head dimension."""
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_hidden_size(self) -> int:
        """Return total hidden dimension used by grouped KV projections."""
        return self.num_key_value_heads * self.head_dim


@dataclass
class SideShape:
    """Shape summary for one query/positive/negative side."""

    count: int = 0
    padded_seq_len: int = 0
    max_effective_len: int = 0
    total_effective_len: int = 0
    total_vision_tokens: int = 0
    vision_segment_tokens: list[int] = field(default_factory=list)


@dataclass
class StepFlops:
    """FLOPs estimate for one simulated optimizer step."""

    config_id: str
    trainer_type: str
    step: int
    per_device_flops: float
    global_flops: float
    grad_cache_phase_a_flops: float
    grad_cache_phase_b_flops: float
    btoks_generation_flops: float
    query_padded_len_max: int
    positive_padded_len_max: int
    negative_padded_len_max: int
    generation_selected: int = 0
    generation_padded_len_max: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowFlops:
    """Aggregated FLOPs estimate for a step window."""

    config_id: str
    trainer_type: str
    start_step: int
    end_step: int
    steps: int
    per_device_gflops: float
    global_gflops: float
    avg_per_device_gflops_per_step: float
    max_per_device_gflops_per_step: float
    btoks_generation_gflops: float
    generation_selected: int


class FlopsEstimator:
    """Static Qwen2-VL FLOPs estimator from tensor shapes."""

    def __init__(
        self,
        config: FlopsConfig,
        *,
        train_forward_backward_multiplier: float = 3.0,
    ) -> None:
        self.config = config
        self.train_forward_backward_multiplier = train_forward_backward_multiplier

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        train_forward_backward_multiplier: float = 3.0,
        trust_remote_code: bool = True,
    ) -> "FlopsEstimator":
        """Build an estimator from a Hugging Face config."""
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        text_config = getattr(hf_config, "text_config", hf_config)
        vision_config = getattr(hf_config, "vision_config", None)
        flops_config = FlopsConfig(
            num_hidden_layers=int(getattr(text_config, "num_hidden_layers")),
            hidden_size=int(getattr(text_config, "hidden_size")),
            intermediate_size=int(getattr(text_config, "intermediate_size")),
            num_attention_heads=int(getattr(text_config, "num_attention_heads")),
            num_key_value_heads=int(getattr(text_config, "num_key_value_heads")),
            vision_depth=int(getattr(vision_config, "depth", 0) or 0),
            vision_embed_dim=int(getattr(vision_config, "embed_dim", 0) or 0),
            vision_num_heads=int(getattr(vision_config, "num_heads", 0) or 0),
            vision_mlp_ratio=float(getattr(vision_config, "mlp_ratio", 4.0) or 4.0),
            vision_spatial_merge_size=int(
                getattr(vision_config, "spatial_merge_size", 2) or 2
            ),
        )
        return cls(
            flops_config,
            train_forward_backward_multiplier=train_forward_backward_multiplier,
        )

    def language_forward_flops(
        self,
        *,
        batch_size: int,
        seq_len: int,
        past_len: int = 0,
    ) -> float:
        """Estimate decoder-only transformer forward FLOPs."""
        if batch_size <= 0 or seq_len <= 0:
            return 0.0
        cfg = self.config
        hidden = cfg.hidden_size
        kv_hidden = cfg.kv_hidden_size
        context_len = past_len + seq_len

        q_proj = cfg.multiply_add_flops * batch_size * seq_len * hidden * hidden
        k_proj = cfg.multiply_add_flops * batch_size * seq_len * hidden * kv_hidden
        v_proj = cfg.multiply_add_flops * batch_size * seq_len * hidden * kv_hidden
        o_proj = cfg.multiply_add_flops * batch_size * seq_len * hidden * hidden
        attention_scores = (
            cfg.multiply_add_flops
            * batch_size
            * cfg.num_attention_heads
            * seq_len
            * context_len
            * cfg.head_dim
        )
        attention_values = attention_scores
        mlp = (
            3
            * cfg.multiply_add_flops
            * batch_size
            * seq_len
            * hidden
            * cfg.intermediate_size
        )
        block = q_proj + k_proj + v_proj + o_proj + attention_scores + attention_values + mlp
        return float(block * cfg.num_hidden_layers)

    def vision_forward_flops(self, *, raw_vision_tokens: int) -> float:
        """Estimate Qwen2-VL vision tower forward FLOPs from raw patch tokens."""
        if raw_vision_tokens <= 0:
            return 0.0
        cfg = self.config
        if cfg.vision_depth <= 0 or cfg.vision_embed_dim <= 0 or cfg.vision_num_heads <= 0:
            return 0.0
        dim = cfg.vision_embed_dim
        heads = cfg.vision_num_heads
        head_dim = dim // heads
        hidden = int(dim * cfg.vision_mlp_ratio)

        qkv = 3 * cfg.multiply_add_flops * raw_vision_tokens * dim * dim
        out = cfg.multiply_add_flops * raw_vision_tokens * dim * dim
        attention = (
            2
            * cfg.multiply_add_flops
            * heads
            * raw_vision_tokens
            * raw_vision_tokens
            * head_dim
        )
        mlp = 2 * cfg.multiply_add_flops * raw_vision_tokens * dim * hidden
        return float((qkv + out + attention + mlp) * cfg.vision_depth)

    def forward_flops_for_side(self, side: SideShape) -> float:
        """Estimate one model forward for a processed side/chunk."""
        vision = sum(
            self.vision_forward_flops(raw_vision_tokens=tokens)
            for tokens in side.vision_segment_tokens
        )
        return self.language_forward_flops(
            batch_size=side.count,
            seq_len=side.padded_seq_len,
        ) + vision

    def generation_forward_flops(
        self,
        *,
        batch_size: int,
        continuation_len: int,
        past_len: int,
        vision_segment_tokens: Sequence[int] = (),
    ) -> float:
        """Estimate continuation forward FLOPs for BToks generation loss."""
        vision = sum(
            self.vision_forward_flops(raw_vision_tokens=tokens)
            for tokens in vision_segment_tokens
        )
        return self.language_forward_flops(
            batch_size=batch_size,
            seq_len=continuation_len,
            past_len=past_len,
        ) + vision


def load_runtime_config(config_path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a config using the same inheritance and override flow as training."""
    loader = ConfigLoader()
    config = loader.load_with_inheritance(str(config_path), resolve_interpolation=False)
    if overrides:
        config = apply_overrides(config, list(overrides))
    OmegaConf.resolve(config)
    return to_native_config(config, resolve=True)


def _backbone_path(config: dict[str, Any]) -> str:
    modules = config["model"]["modules"]
    for module_config in modules.values():
        if isinstance(module_config, dict) and "model_name_or_path" in module_config:
            return str(module_config["model_name_or_path"])
    raise ValueError("Model config does not contain a backbone model_name_or_path")


def _btoks_token_count(config: dict[str, Any]) -> int:
    modules = config.get("model", {}).get("modules", {})
    injector = modules.get("injector", {}) if isinstance(modules, dict) else {}
    if isinstance(injector, dict):
        return int(injector.get("num_tokens", 0) or 0)
    return 0


def _fast_decode_image(data: bytes) -> Image.Image:
    """Open image metadata lazily without RGB conversion for shape simulation."""
    return Image.open(BytesIO(data))


def _patch_dataset_decode_image_for_shape_only() -> None:
    """Patch dataset decode helpers so shape-only scans do not decode pixels."""
    import vlm2emb.data.datasets.const as const

    const.decode_image = _fast_decode_image
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("vlm2emb.data.datasets"):
            continue
        if hasattr(module, "decode_image"):
            setattr(module, "decode_image", _fast_decode_image)


def _build_runtime(config: dict[str, Any], *, shape_only: bool = True) -> tuple[Any, Any, Any]:
    """Build dataset, collator, and processor wrapper without building a model."""
    import vlm2emb.data.collators  # noqa: F401
    import vlm2emb.data.datasets  # noqa: F401
    import vlm2emb.data.processors  # noqa: F401

    if shape_only:
        _patch_dataset_decode_image_for_shape_only()

    train_config = config["train"]
    processor_config = dict(config["processor"])
    wrapper = AutoProcessorWrapper.from_config(processor_config)
    dataset = AutoDataset.from_config(train_config["dataset"])
    collator = AutoCollator.from_config(train_config["collator"], wrapper=wrapper)
    return dataset, collator, wrapper


def _sample_len(sample: dict[str, Any] | None) -> int:
    if sample is None:
        return 0
    input_length = sample.get("input_length")
    if input_length is not None:
        return int(input_length)
    input_ids = sample.get("input_ids")
    if input_ids is None:
        return 0
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    if isinstance(input_ids, (list, tuple)):
        if len(input_ids) == 1 and isinstance(input_ids[0], (list, tuple)):
            return len(input_ids[0])
        return len(input_ids)
    return 1


def _vision_token_segments(sample: dict[str, Any] | None) -> list[int]:
    if sample is None:
        return []
    if "vision_segment_tokens" in sample:
        return [int(value) for value in sample.get("vision_segment_tokens") or []]
    segments: list[int] = []
    for key in ("image_grid_thw", "video_grid_thw"):
        grid = sample.get(key)
        if grid is None:
            continue
        if torch.is_tensor(grid):
            rows = grid.detach().cpu().view(-1, 3).tolist()
        else:
            rows = grid
        for row in rows:
            if len(row) >= 3:
                segments.append(int(row[0]) * int(row[1]) * int(row[2]))
    return segments


def _processor_size(processor: Any) -> tuple[int, int]:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        shortest = int(size.get("shortest_edge", getattr(processor, "min_pixels", 56 * 56)))
        longest = int(size.get("longest_edge", getattr(processor, "max_pixels", 14 * 14 * 4 * 1280)))
        return shortest, longest
    return (
        int(getattr(processor, "min_pixels", 56 * 56)),
        int(getattr(processor, "max_pixels", 14 * 14 * 4 * 1280)),
    )


def _grid_for_image(image: Any, processor: Any) -> tuple[int, int, int]:
    width, height = image.size
    patch_size = int(getattr(processor, "patch_size", 14))
    merge_size = int(getattr(processor, "merge_size", 2))
    min_pixels, max_pixels = _processor_size(processor)
    resized_h, resized_w = smart_resize(
        int(height),
        int(width),
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return (1, resized_h // patch_size, resized_w // patch_size)


def _grid_for_video(frames: Sequence[Any], processor: Any) -> tuple[int, int, int]:
    if not frames:
        return (0, 0, 0)
    width, height = frames[0].size
    patch_size = int(getattr(processor, "patch_size", 14))
    merge_size = int(getattr(processor, "merge_size", 2))
    temporal_patch_size = int(getattr(processor, "temporal_patch_size", 2))
    min_pixels, max_pixels = _processor_size(processor)
    resized_h, resized_w = smart_resize(
        int(height),
        int(width),
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    grid_t = math.ceil(len(frames) / temporal_patch_size)
    return (grid_t, resized_h // patch_size, resized_w // patch_size)


def _expanded_visual_text(
    text: str,
    *,
    image_token: str,
    video_token: str,
    image_grids: Sequence[tuple[int, int, int]],
    video_grids: Sequence[tuple[int, int, int]],
    merge_size: int,
) -> str:
    merge_length = merge_size**2
    expanded = text
    for grid in image_grids:
        if image_token not in expanded:
            break
        n_tokens = max(1, (grid[0] * grid[1] * grid[2]) // merge_length)
        expanded = expanded.replace(image_token, image_token * n_tokens, 1)
    for grid in video_grids:
        if video_token not in expanded:
            break
        n_tokens = max(1, (grid[0] * grid[1] * grid[2]) // merge_length)
        expanded = expanded.replace(video_token, video_token * n_tokens, 1)
    return expanded


def _shape_process_payload(payload: dict[str, Any], wrapper: Any) -> dict[str, Any]:
    """Return token/grid shape for one payload without pixel preprocessing."""
    text, media, media_metadata = extract_multimodal_payload(
        payload,
        wrapper_name=type(wrapper).__name__,
        context="Training",
    )
    normalize = getattr(wrapper, "_normalize_visual_text", None)
    if callable(normalize):
        text = normalize(text)

    image_token = getattr(wrapper, "IMAGE_TOKEN", "<|image_pad|>")
    video_token = getattr(wrapper, "VIDEO_TOKEN", "<|video_pad|>")
    image_processor = wrapper.processor.image_processor
    video_processor = getattr(wrapper.processor, "video_processor", None) or image_processor
    image_merge = int(getattr(image_processor, "merge_size", 2))
    video_merge = int(getattr(video_processor, "merge_size", image_merge))

    image_grids: list[tuple[int, int, int]] = []
    video_grids: list[tuple[int, int, int]] = []
    if media:
        if video_token in text:
            video_grids.append(_grid_for_video(media, video_processor))
        else:
            image_grids.extend(_grid_for_image(image, image_processor) for image in media)

    expanded_text = _expanded_visual_text(
        text or " ",
        image_token=image_token,
        video_token=video_token,
        image_grids=image_grids,
        video_grids=video_grids,
        merge_size=image_merge if image_grids else video_merge,
    )
    tokenized = wrapper.tokenizer(
        expanded_text,
        max_length=getattr(wrapper, "max_length", None),
        truncation=bool(getattr(wrapper, "max_length", None)) and not (image_grids or video_grids),
        add_special_tokens=True,
    )
    input_ids = tokenized["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_length = len(input_ids[0])
    else:
        input_length = len(input_ids)

    vision_segments = [
        grid_t * grid_h * grid_w
        for grid_t, grid_h, grid_w in [*image_grids, *video_grids]
    ]
    result: dict[str, Any] = {
        "input_length": input_length,
        "vision_segment_tokens": vision_segments,
    }
    if media_metadata is not None:
        result["media_metadata"] = media_metadata
    return result


def _shape_collate(batch: list[dict[str, Any]], collator: Any, wrapper: Any) -> dict[str, Any]:
    """Shape-only equivalent of TrainingCollator for FLOPs simulation."""
    outputs: dict[str, Any] = {}
    instruction_policy = getattr(collator, "instruction_policy", None)
    for field_name in ("query", "positive", "negative"):
        values: list[dict[str, Any] | None] = []
        for sample in batch:
            field_data = sample.get(field_name, {})
            text = field_data.get("text", "") or ""
            if field_name == "negative" and not text:
                values.append(None)
                continue
            rendered = render_instruction_text(
                text,
                field_name=field_name,
                instruction_policy=instruction_policy,
            )
            payload = _build_train_field_payload(
                field_data,
                sample,
                field_name=field_name,
                text=rendered or " ",
            )
            values.append(_shape_process_payload(payload, wrapper))
        outputs[field_name] = values

    metadata_field_names: set[str] = set()
    for sample in batch:
        metadata = sample.get("metadata")
        if isinstance(metadata, dict):
            metadata_field_names.update(metadata.keys())
    if metadata_field_names:
        outputs["metadata"] = {
            key: [
                sample.get("metadata", {}).get(key)
                if isinstance(sample.get("metadata", {}), dict)
                else None
                for sample in batch
            ]
            for key in sorted(metadata_field_names)
        }
    return outputs


def _side_shape(samples: Sequence[dict[str, Any] | None], *, extra_tokens: int = 0) -> SideShape:
    lengths = [_sample_len(sample) + extra_tokens for sample in samples if sample is not None]
    if not lengths:
        return SideShape()
    vision_segments = [
        tokens
        for sample in samples
        for tokens in _vision_token_segments(sample)
    ]
    return SideShape(
        count=len(lengths),
        padded_seq_len=max(lengths),
        max_effective_len=max(lengths),
        total_effective_len=sum(lengths),
        total_vision_tokens=sum(vision_segments),
        vision_segment_tokens=vision_segments,
    )


def _chunked(items: Sequence[Any], chunk_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def _sort_pair_by_query_length(
    query_samples: Sequence[Any],
    positive_samples: Sequence[Any],
) -> tuple[list[Any], list[Any]]:
    order = sorted(
        range(len(query_samples)),
        key=lambda idx: (_sample_len(query_samples[idx]), idx),
    )
    return [query_samples[idx] for idx in order], [positive_samples[idx] for idx in order]


def _metadata_modes(metadata: dict[str, Any] | None, batch_size: int) -> list[str]:
    if not isinstance(metadata, dict):
        return ["none"] * batch_size
    raw = metadata.get("ntp_side")
    if not isinstance(raw, (list, tuple)):
        return ["none"] * batch_size
    modes: list[str] = []
    for index in range(batch_size):
        value = str(raw[index] if index < len(raw) else "none").strip().lower()
        if value in {"query", "qry"}:
            modes.append("query")
        elif value in {"positive", "pos", "target", "tgt"}:
            modes.append("positive")
        elif value in {"both", "all"}:
            modes.append("both")
        else:
            modes.append("none")
    return modes


def _decorate_lengths(
    samples: Sequence[dict[str, Any]],
    *,
    prefix_len: int,
    suffix_len: int,
) -> SideShape:
    lengths = [_sample_len(sample) + prefix_len + suffix_len for sample in samples]
    if not lengths:
        return SideShape()
    vision_segments = [
        tokens
        for sample in samples
        for tokens in _vision_token_segments(sample)
    ]
    return SideShape(
        count=len(lengths),
        padded_seq_len=max(lengths),
        max_effective_len=max(lengths),
        total_effective_len=sum(lengths),
        total_vision_tokens=sum(vision_segments),
        vision_segment_tokens=vision_segments,
    )


def _estimate_vlm2vec_step(
    *,
    estimator: FlopsEstimator,
    config_id: str,
    trainer_type: str,
    step: int,
    query_samples: list[dict[str, Any]],
    positive_samples: list[dict[str, Any]],
    negative_samples: list[dict[str, Any] | None],
    gc_chunk_size: int,
    world_size: int,
) -> StepFlops:
    sorted_query, sorted_positive = _sort_pair_by_query_length(query_samples, positive_samples)
    phase_a = 0.0
    phase_b_forward = 0.0
    max_query_len = 0
    max_positive_len = 0
    for query_chunk, positive_chunk in zip(
        _chunked(sorted_query, gc_chunk_size),
        _chunked(sorted_positive, gc_chunk_size),
        strict=True,
    ):
        query_shape = _side_shape(query_chunk)
        positive_shape = _side_shape(positive_chunk)
        max_query_len = max(max_query_len, query_shape.padded_seq_len)
        max_positive_len = max(max_positive_len, positive_shape.padded_seq_len)
        query_forward = estimator.forward_flops_for_side(query_shape)
        positive_forward = estimator.forward_flops_for_side(positive_shape)
        phase_a += query_forward + positive_forward
        phase_b_forward += query_forward + positive_forward

    phase_b = phase_b_forward * estimator.train_forward_backward_multiplier
    per_device = phase_a + phase_b
    negative_shape = _side_shape(negative_samples)
    return StepFlops(
        config_id=config_id,
        trainer_type=trainer_type,
        step=step,
        per_device_flops=per_device,
        global_flops=per_device * world_size,
        grad_cache_phase_a_flops=phase_a,
        grad_cache_phase_b_flops=phase_b,
        btoks_generation_flops=0.0,
        query_padded_len_max=max_query_len,
        positive_padded_len_max=max_positive_len,
        negative_padded_len_max=negative_shape.padded_seq_len,
        metadata={"estimate_method": "qwen2vl_static_shape_v1"},
    )


def _estimate_btoks_step(
    *,
    estimator: FlopsEstimator,
    config: dict[str, Any],
    config_id: str,
    trainer_type: str,
    step: int,
    query_samples: list[dict[str, Any]],
    positive_samples: list[dict[str, Any]],
    negative_samples: list[dict[str, Any] | None],
    metadata: dict[str, Any] | None,
    wrapper: Any,
    gc_chunk_size: int,
    world_size: int,
) -> StepFlops:
    train_args = config["train"]["args"]
    btoks_tokens = _btoks_token_count(config)
    generation_weight = float(train_args.get("generation_loss_weight", 0.0) or 0.0)
    generation_duration = int(train_args.get("generation_loss_duration", 0) or 0)
    generation_enabled = generation_weight > 0 and step <= generation_duration
    generation_kv_mode = str(train_args.get("generation_kv_mode", "compressed")).lower()
    prefix_ids = wrapper.tokenizer(
        str(train_args.get("generation_prefix_text", "<|im_start|>assistant\n")),
        add_special_tokens=False,
    )["input_ids"]
    suffix_ids = wrapper.tokenizer(
        str(train_args.get("generation_suffix_text", "<|im_end|>")),
        add_special_tokens=False,
    )["input_ids"]
    prefix_len = len(prefix_ids)
    suffix_len = len(suffix_ids)

    modes = _metadata_modes(metadata, len(query_samples))
    query_group: list[dict[str, Any]] = []
    positive_group: list[dict[str, Any]] = []
    for idx, mode in enumerate(modes):
        query_group.append(
            {
                "self": query_samples[idx],
                "opposite": positive_samples[idx],
                "should_generate": mode in {"positive", "both"},
            }
        )
        positive_group.append(
            {
                "self": positive_samples[idx],
                "opposite": query_samples[idx],
                "should_generate": mode in {"query", "both"},
            }
        )

    order = sorted(
        range(len(query_group)),
        key=lambda idx: (_sample_len(query_group[idx]["self"]), idx),
    )
    sorted_query = [query_group[idx] for idx in order]
    sorted_positive = [positive_group[idx] for idx in order]

    phase_a = 0.0
    phase_b_forward = 0.0
    generation_forward = 0.0
    selected_total = 0
    max_query_len = 0
    max_positive_len = 0
    max_generation_len = 0

    for query_chunk, positive_chunk in zip(
        _chunked(sorted_query, gc_chunk_size),
        _chunked(sorted_positive, gc_chunk_size),
        strict=True,
    ):
        for group_name, chunk in (("query", query_chunk), ("positive", positive_chunk)):
            self_samples = [sample["self"] for sample in chunk]
            self_shape = _side_shape(self_samples, extra_tokens=btoks_tokens)
            if group_name == "query":
                max_query_len = max(max_query_len, self_shape.padded_seq_len)
            else:
                max_positive_len = max(max_positive_len, self_shape.padded_seq_len)
            self_forward = estimator.forward_flops_for_side(self_shape)
            phase_a += self_forward
            phase_b_forward += self_forward

            if generation_enabled:
                selected = [sample for sample in chunk if sample["should_generate"]]
                if selected:
                    selected_total += len(selected)
                    opposite_samples = [sample["opposite"] for sample in selected]
                    opposite_shape = _decorate_lengths(
                        opposite_samples,
                        prefix_len=prefix_len,
                        suffix_len=suffix_len,
                    )
                    max_generation_len = max(
                        max_generation_len,
                        opposite_shape.padded_seq_len,
                    )
                    if generation_kv_mode == "full":
                        past_len = self_shape.padded_seq_len
                    else:
                        past_len = btoks_tokens if btoks_tokens > 0 else 1
                    generation_forward += estimator.generation_forward_flops(
                        batch_size=opposite_shape.count,
                        continuation_len=opposite_shape.padded_seq_len,
                        past_len=past_len,
                        vision_segment_tokens=opposite_shape.vision_segment_tokens,
                    )

    phase_b = phase_b_forward * estimator.train_forward_backward_multiplier
    generation = generation_forward * estimator.train_forward_backward_multiplier
    per_device = phase_a + phase_b + generation
    negative_shape = _side_shape(negative_samples)
    return StepFlops(
        config_id=config_id,
        trainer_type=trainer_type,
        step=step,
        per_device_flops=per_device,
        global_flops=per_device * world_size,
        grad_cache_phase_a_flops=phase_a,
        grad_cache_phase_b_flops=phase_b,
        btoks_generation_flops=generation,
        query_padded_len_max=max_query_len,
        positive_padded_len_max=max_positive_len,
        negative_padded_len_max=negative_shape.padded_seq_len,
        generation_selected=selected_total,
        generation_padded_len_max=max_generation_len,
        metadata={
            "estimate_method": "qwen2vl_static_shape_v1",
            "generation_enabled": generation_enabled,
            "generation_kv_mode": generation_kv_mode,
            "btoks_tokens": btoks_tokens,
        },
    )


def _sampler_indices(
    dataset: Any,
    train_args: dict[str, Any],
    *,
    per_device_batch_size: int,
    seed: int,
    epoch: int,
) -> list[int]:
    interleave_batch_size = int(train_args.get("interleave_batch_size", 0) or 0)
    if interleave_batch_size > 0 and hasattr(dataset, "dataset_lengths"):
        sampler = BatchInterleaveSampler(
            dataset=dataset,
            batch_size=interleave_batch_size,
            stopping_strategy=train_args.get("interleave_stopping_strategy", "all_exhausted"),
            shuffle_within_dataset=train_args.get("interleave_shuffle_within_dataset", True),
            seed=seed,
            per_device_batch_size=None,
            num_processes=None,
        )
        sampler.set_epoch(epoch)
        return list(sampler)

    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(len(dataset), generator=generator).tolist()


def _iter_batches(
    dataset: Any,
    train_args: dict[str, Any],
    *,
    per_device_batch_size: int,
    max_steps: int,
    world_size: int,
    rank: int,
    drop_last: bool,
) -> Iterator[tuple[list[Any], str | None]]:
    """Yield simulated per-rank batches, cycling epochs when needed."""
    seed = int(train_args.get("seed", 42) or 42)
    step = 0
    epoch = 0
    global_batch_size = per_device_batch_size * world_size
    batch_getitems = getattr(dataset, "__getitems__", None)
    while step < max_steps:
        indices = _sampler_indices(
            dataset,
            train_args,
            per_device_batch_size=per_device_batch_size,
            seed=seed,
            epoch=epoch,
        )
        for start in range(0, len(indices), global_batch_size):
            global_indices = indices[start : start + global_batch_size]
            if len(global_indices) < global_batch_size and drop_last:
                continue
            rank_indices = global_indices[
                rank * per_device_batch_size : (rank + 1) * per_device_batch_size
            ]
            if len(rank_indices) < per_device_batch_size and drop_last:
                continue
            if callable(batch_getitems):
                batch = list(batch_getitems(rank_indices))
            else:
                batch = [dataset[index] for index in rank_indices]
            source_name = None
            if rank_indices and hasattr(dataset, "_get_dataset_and_local_index"):
                ds_idx, _ = dataset._get_dataset_and_local_index(rank_indices[0])
                names = getattr(dataset, "names", None)
                source_name = str(names[ds_idx]) if names else str(ds_idx)
            yield batch, source_name
            step += 1
            if step >= max_steps:
                break
        epoch += 1


def simulate_config(
    config_path: str | Path,
    *,
    options: SimulationOptions | None = None,
) -> list[StepFlops]:
    """Simulate one training config and return step-level FLOPs estimates."""
    opts = options or SimulationOptions()
    config = load_runtime_config(config_path, opts.overrides)
    config_id = Path(config_path).stem
    train_args = config["train"]["args"]
    trainer_type = str(config["train"]["trainer"]["type"])
    per_device_batch_size = int(
        opts.batch_size or train_args.get("per_device_train_batch_size", 1)
    )
    drop_last = (
        bool(train_args.get("dataloader_drop_last", False))
        if opts.drop_last is None
        else bool(opts.drop_last)
    )
    if opts.world_size <= 0:
        raise ValueError("world_size must be positive")
    if opts.rank < 0 or opts.rank >= opts.world_size:
        raise ValueError("rank must be in [0, world_size)")

    dataset, collator, wrapper = _build_runtime(config, shape_only=not opts.exact_processor)
    estimator = FlopsEstimator.from_pretrained(
        _backbone_path(config),
        train_forward_backward_multiplier=opts.train_forward_backward_multiplier,
        trust_remote_code=True,
    )
    gc_chunk_size = int(train_args.get("gc_chunk_size", per_device_batch_size) or per_device_batch_size)

    steps: list[StepFlops] = []
    shape_cache: dict[str, dict[str, Any]] = {}
    for step_idx, (raw_batch, source_name) in enumerate(
        _iter_batches(
            dataset,
            train_args,
            per_device_batch_size=per_device_batch_size,
            max_steps=opts.max_steps,
            world_size=opts.world_size,
            rank=opts.rank,
            drop_last=drop_last,
        ),
        start=1,
    ):
        cache_key = source_name if opts.cache_source_shapes and source_name else None
        if cache_key is not None and cache_key in shape_cache:
            batch = shape_cache[cache_key]
        else:
            batch = (
                collator(raw_batch)
                if opts.exact_processor
                else _shape_collate(raw_batch, collator, wrapper)
            )
            if cache_key is not None:
                shape_cache[cache_key] = batch
        query_samples = list(batch["query"])
        positive_samples = list(batch["positive"])
        negative_samples = list(batch.get("negative", []))
        if trainer_type in {"btoks_trainer", "btoks"}:
            step = _estimate_btoks_step(
                estimator=estimator,
                config=config,
                config_id=config_id,
                trainer_type=trainer_type,
                step=step_idx,
                query_samples=query_samples,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                metadata=batch.get("metadata"),
                wrapper=wrapper,
                gc_chunk_size=gc_chunk_size,
                world_size=opts.world_size,
            )
        else:
            step = _estimate_vlm2vec_step(
                estimator=estimator,
                config_id=config_id,
                trainer_type=trainer_type,
                step=step_idx,
                query_samples=query_samples,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                gc_chunk_size=gc_chunk_size,
                world_size=opts.world_size,
            )
        steps.append(step)
        if opts.progress_every > 0 and step_idx % opts.progress_every == 0:
            logger.info(
                "%s: simulated %d/%d steps",
                config_id,
                step_idx,
                opts.max_steps,
            )
    return steps


def aggregate_windows(steps: Sequence[StepFlops], window_steps: int) -> list[WindowFlops]:
    """Aggregate step-level results into fixed-size windows."""
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    windows: list[WindowFlops] = []
    for start in range(0, len(steps), window_steps):
        chunk = list(steps[start : start + window_steps])
        if not chunk:
            continue
        per_device = sum(step.per_device_flops for step in chunk)
        global_flops = sum(step.global_flops for step in chunk)
        generation = sum(step.btoks_generation_flops for step in chunk)
        windows.append(
            WindowFlops(
                config_id=chunk[0].config_id,
                trainer_type=chunk[0].trainer_type,
                start_step=chunk[0].step,
                end_step=chunk[-1].step,
                steps=len(chunk),
                per_device_gflops=per_device / FLOPS_PER_GIGA,
                global_gflops=global_flops / FLOPS_PER_GIGA,
                avg_per_device_gflops_per_step=(per_device / len(chunk)) / FLOPS_PER_GIGA,
                max_per_device_gflops_per_step=max(
                    step.per_device_flops for step in chunk
                )
                / FLOPS_PER_GIGA,
                btoks_generation_gflops=generation / FLOPS_PER_GIGA,
                generation_selected=sum(step.generation_selected for step in chunk),
            )
        )
    return windows


def write_jsonl(path: str | Path, records: Sequence[Any]) -> None:
    """Write dataclass records as JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def format_windows(windows: Sequence[WindowFlops]) -> str:
    """Return a compact console table for window records."""
    lines = [
        "config_id\tsteps\tglobal_GFLOPs\tper_device_GFLOPs\tavg_per_step_GFLOPs\tmax_step_GFLOPs\tgeneration_GFLOPs"
    ]
    for window in windows:
        lines.append(
            "\t".join(
                [
                    window.config_id,
                    f"{window.start_step}-{window.end_step}",
                    f"{window.global_gflops:.6f}",
                    f"{window.per_device_gflops:.6f}",
                    f"{window.avg_per_device_gflops_per_step:.6f}",
                    f"{window.max_per_device_gflops_per_step:.6f}",
                    f"{window.btoks_generation_gflops:.6f}",
                ]
            )
        )
    return "\n".join(lines)


def summarize_total(steps: Sequence[StepFlops]) -> dict[str, float]:
    """Return total FLOPs summary for one config."""
    if not steps:
        return {"steps": 0.0, "global_tflops": math.nan, "per_device_tflops": math.nan}
    return {
        "steps": float(len(steps)),
        "global_tflops": sum(step.global_flops for step in steps) / FLOPS_PER_TERA,
        "per_device_tflops": sum(step.per_device_flops for step in steps) / FLOPS_PER_TERA,
    }
