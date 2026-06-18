"""Qwen3-VL series processor wrapper.

This module provides Qwen3VLProcessorWrapper that wraps Qwen3-VL processors
to provide unified input/output interface with token-based routing.

Design:
    - ProcessorWrapper returns List[Dict], one dict per sample (not batched)
    - Batching/concatenation is deferred to trainer for flexibility
    - Use batch_processor_outputs() to batch samples before model forward

Example:
    >>> from vlm2emb.data.processors import Qwen3VLProcessorWrapper, batch_processor_outputs
    >>> wrapper = Qwen3VLProcessorWrapper.from_config({
    ...     "type": "qwen3_vl",
    ...     "processor_name": "Qwen/Qwen3-VL-2B-Instruct",
    ... })
    >>> # Returns List[Dict], one per sample
    >>> samples = wrapper(texts, images)
    >>> # Batch before model forward
    >>> batched = batch_processor_outputs(samples, wrapper)
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import ChannelDimension

from vlm2emb.auto import AutoProcessorWrapper
from vlm2emb.data.processors.media import extract_multimodal_payload, extract_retrieval_payload

# Use standard logging so errors remain visible from all processes.
logger = logging.getLogger(__name__)


# Qwen2-VL and Qwen3-VL wrappers intentionally stay separate until their
# processor contracts are stable enough to share a base class without hiding
# model-family-specific behavior.


@AutoProcessorWrapper.register("qwen3_vl")
class Qwen3VLProcessorWrapper:
    """ProcessorWrapper for Qwen3-VL series (Qwen3-VL).

    Handles:
    - Token-based routing: <|image_pad|> for images, <|video_pad|> for videos
    - Returns List[Dict] for flexible batching in trainer
    - Image resizing handled by processor's smart_resize (preserves aspect ratio)

    Args:
        processor: Qwen3-VL processor instance.
        max_length: Maximum sequence length for text-only inputs.

    Example:
        >>> wrapper = Qwen3VLProcessorWrapper(processor, max_length=512)
        >>> samples = wrapper(
        ...     texts=["<|image_pad|> Describe this image"],
        ...     images=[[pil_image]],
        ... )
        >>> # samples is List[Dict], one per input
        >>> batched = batch_processor_outputs(samples, wrapper)
    """

    IMAGE_TOKEN = "<|image_pad|>"
    VIDEO_TOKEN = "<|video_pad|>"
    VISION_START_TOKEN = "<|vision_start|>"
    VISION_END_TOKEN = "<|vision_end|>"
    IMAGE_SEGMENT = f"{VISION_START_TOKEN}{IMAGE_TOKEN}{VISION_END_TOKEN}"
    VIDEO_SEGMENT = f"{VISION_START_TOKEN}{VIDEO_TOKEN}{VISION_END_TOKEN}"
    LEGACY_IMAGE_TOKENS = ("<|image_1|>", "<image>")
    LEGACY_VIDEO_TOKENS = ("<video>",)

    def __init__(
        self,
        processor: Any,
        max_length: int | None = None,
        wrap_visual_tokens_with_boundaries: bool = True,
    ):
        """Initialize wrapper.

        Args:
            processor: Qwen3-VL processor from transformers.
            max_length: Max sequence length for text-only inputs.
            wrap_visual_tokens_with_boundaries: Whether to expand plain visual
                        placeholders to the explicit Qwen boundary form before
                        processor tokenization. Default stays enabled so
                        production behavior matches the currently supported
                        checkpoint/runtime contract.
        """
        self.processor = processor
        self.max_length = max_length
        self.wrap_visual_tokens_with_boundaries = wrap_visual_tokens_with_boundaries
        self._input_data_format = self._detect_input_data_format()
        self._video_size = self._get_video_size()

    @property
    def tokenizer(self):
        """Access the underlying tokenizer."""
        return cast(Any, self.processor).tokenizer

    @property
    def vision_token_ids(self) -> list[int]:
        """Return vision-related special token IDs from the underlying processor."""
        ids = []
        for attr in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            tid = getattr(self.processor, attr, None)
            if isinstance(tid, int):
                ids.append(tid)
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            for token in (
                self.IMAGE_TOKEN,
                self.VIDEO_TOKEN,
                self.VISION_START_TOKEN,
                self.VISION_END_TOKEN,
            ):
                tid = tokenizer.convert_tokens_to_ids(token)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
        ids = list(dict.fromkeys(ids))
        return ids

    def _normalize_visual_text(self, text: str) -> str:
        """Canonicalize visual placeholders to the Qwen vision boundary format.

        The optional switch is reserved for regression experiments. Default
        production behavior keeps the explicit boundary tokens.
        """
        normalized = text or ""
        for legacy_token in self.LEGACY_IMAGE_TOKENS:
            normalized = normalized.replace(legacy_token, self.IMAGE_TOKEN)
        for legacy_token in self.LEGACY_VIDEO_TOKENS:
            normalized = normalized.replace(legacy_token, self.VIDEO_TOKEN)
        if not self.wrap_visual_tokens_with_boundaries:
            return normalized

        img_placeholder = "__VLM2EMB_IMAGE_SEGMENT__"
        vid_placeholder = "__VLM2EMB_VIDEO_SEGMENT__"
        normalized = re.sub(
            rf"{re.escape(self.VISION_START_TOKEN)}\s*{re.escape(self.IMAGE_TOKEN)}\s*{re.escape(self.VISION_END_TOKEN)}",
            img_placeholder,
            normalized,
        )
        normalized = re.sub(
            rf"{re.escape(self.VISION_START_TOKEN)}\s*{re.escape(self.VIDEO_TOKEN)}\s*{re.escape(self.VISION_END_TOKEN)}",
            vid_placeholder,
            normalized,
        )
        normalized = normalized.replace(self.IMAGE_TOKEN, self.IMAGE_SEGMENT)
        normalized = normalized.replace(self.VIDEO_TOKEN, self.VIDEO_SEGMENT)
        normalized = normalized.replace(img_placeholder, self.IMAGE_SEGMENT)
        normalized = normalized.replace(vid_placeholder, self.VIDEO_SEGMENT)
        return normalized

    def _detect_input_data_format(self) -> ChannelDimension:
        """Detect the correct input_data_format based on image processor type.

        Fast processor (pil_to_tensor) produces (C, H, W) = channels_first.
        Slow processor (to_numpy_array) produces (H, W, C) = channels_last.
        Explicitly passing this avoids the ambiguous channel dimension warning
        from infer_channel_dimension_format when images have H or W in {1, 3}.
        """
        try:
            from transformers.image_processing_utils_fast import (
                BaseImageProcessorFast,
            )

            is_fast = isinstance(cast(Any, self.processor).image_processor, BaseImageProcessorFast)
        except ImportError:
            is_fast = False
        fmt = ChannelDimension.FIRST if is_fast else ChannelDimension.LAST
        logger.debug(
            "Detected image processor: %s -> input_data_format=%s",
            type(cast(Any, self.processor).image_processor).__name__,
            fmt,
        )
        return fmt

    def _get_video_size(self) -> dict[str, int] | None:
        """Return video-specific size if different from image, else None.

        Note: Qwen3VL's video_processor uses size dict (shortest_edge/longest_edge)
        instead of min_pixels/max_pixels attributes.
        """
        vid_proc = getattr(self.processor, "video_processor", None)
        if vid_proc is None:
            return None
        img_proc = cast(Any, self.processor).image_processor
        vid_size = vid_proc.size
        img_min = img_proc.min_pixels
        img_max = img_proc.max_pixels
        vid_min = vid_size.get("shortest_edge", img_min)
        vid_max = vid_size.get("longest_edge", img_max)
        if vid_max != img_max or vid_min != img_min:
            return {"shortest_edge": vid_min, "longest_edge": vid_max}
        return None

    def _get_video_temporal_factor(self) -> int:
        """Return the minimum frame factor required by Qwen3 video patches."""
        vid_proc = getattr(self.processor, "video_processor", None)
        temporal_factor = getattr(vid_proc, "temporal_patch_size", 2)
        if not isinstance(temporal_factor, int) or temporal_factor < 1:
            return 2
        return temporal_factor

    def _ensure_video_temporal_factor(
        self,
        media: list[Image.Image],
        timing_metadata: dict[str, Any],
    ) -> list[Image.Image]:
        """Repeat the last frame when Qwen3-VL receives too few frames."""
        temporal_factor = self._get_video_temporal_factor()
        if len(media) >= temporal_factor:
            return media
        if not media:
            return media

        padded_media = list(media)
        padded_media.extend([media[-1]] * (temporal_factor - len(media)))

        frame_indices = timing_metadata.get("frames_indices")
        if isinstance(frame_indices, list):
            if not frame_indices:
                frame_indices = [0]
            missing_indices = max(0, temporal_factor - len(frame_indices))
            timing_metadata["frames_indices"] = frame_indices + [frame_indices[-1]] * missing_indices
        return padded_media

    def save_pretrained(self, *args, **kwargs):
        """Delegate to underlying processor for checkpoint saving."""
        return self.processor.save_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Qwen3VLProcessorWrapper:
        """Create wrapper from config dict.

        Args:
            config: Configuration dict. Must contain 'processor_name' or 'processor'.

        Config example:
            type: qwen3_vl
            processor_name: Qwen/Qwen3-VL-2B-Instruct
            max_length: 512
            image_min_pixels: 3136      # 28*28*4
            image_max_pixels: 1003520   # 28*28*1280
            video_min_pixels: 3136      # defaults to image_min_pixels
            video_max_pixels: 602112    # 28*28*768, defaults to 602112

        Backward compatible - old min_pixels/max_pixels still work (applied to both).

        Returns:
            Initialized Qwen3VLProcessorWrapper.
        """
        config = config.copy()
        config.pop("type", None)

        processor_name = config.pop("processor_name", None)
        processor = config.pop("processor", None)

        # New params: image_*/video_* with backward compat for old min_pixels/max_pixels.
        old_min = config.pop("min_pixels", 28 * 28 * 4)
        old_max = config.pop("max_pixels", 28 * 28 * 1280)
        image_min_pixels = config.pop("image_min_pixels", old_min)
        image_max_pixels = config.pop("image_max_pixels", old_max)
        video_min_pixels = config.pop("video_min_pixels", image_min_pixels)
        video_max_pixels = config.pop("video_max_pixels", 28 * 28 * 768)  # 602112

        if processor is None:
            if processor_name is None:
                raise ValueError("Either 'processor' or 'processor_name' required in config")
            # Load with image resolution — AutoProcessor applies same kwargs to all sub-processors.
            # See: https://github.com/QwenLM/Qwen3-VL/issues/1052
            processor = AutoProcessor.from_pretrained(
                processor_name,
                trust_remote_code=True,
                min_pixels=image_min_pixels,
                max_pixels=image_max_pixels,
                size={"shortest_edge": image_min_pixels, "longest_edge": image_max_pixels},
            )

        # Override video_processor if different from image. Qwen3VL video_processor
        # uses size dict, not min_pixels/max_pixels attributes.
        vid_proc = getattr(processor, "video_processor", None)
        if vid_proc is not None and (
            video_min_pixels != image_min_pixels or video_max_pixels != image_max_pixels
        ):
            vid_proc.size = {
                "shortest_edge": video_min_pixels,
                "longest_edge": video_max_pixels,
            }

        return cls(processor=processor, **config)

    def __call__(
        self,
        texts: list[str],
        images: list[list[Image.Image] | None],
        media_metadata: list[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, torch.Tensor | None]]:
        """Process batch of text-image pairs, returning List[Dict].

        Each sample is processed independently and returned as a separate dict.
        Use batch_processor_outputs() to batch samples before model forward.
        """
        results: list[dict[str, torch.Tensor | None]] = []
        effective_media_metadata = media_metadata or [None] * len(texts)

        for text, imgs, sample_media_metadata in zip(texts, images, effective_media_metadata, strict=True):
            result = self._process_single(text, imgs, sample_media_metadata)

            sample = {
                "input_ids": result["input_ids"],
                "attention_mask": result.get(
                    "attention_mask",
                    torch.ones_like(result["input_ids"]),
                ),
                "pixel_values": result.get("pixel_values"),
                "image_grid_thw": result.get("image_grid_thw"),
                "pixel_values_videos": result.get("pixel_values_videos"),
                "video_grid_thw": result.get("video_grid_thw"),
                "mm_token_type_ids": result.get("mm_token_type_ids"),
            }
            results.append(sample)

        return results

    def process_multimodal_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor | None]]:
        """Process one batch of direct ``{"text", "media"}`` runtime payloads."""
        texts: list[str] = []
        images: list[list[Image.Image] | None] = []
        media_metadata: list[dict[str, Any] | None] = []
        for payload in batch:
            text, sample_images, sample_media_metadata = extract_multimodal_payload(
                payload,
                wrapper_name=type(self).__name__,
                context="Training",
            )
            texts.append(text)
            images.append(sample_images)
            media_metadata.append(sample_media_metadata)
        return self(texts, images, media_metadata=media_metadata)

    def process_retrieval_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor | None]]:
        """Process one retrieval-item batch emitted by the new runtime schema."""

        texts: list[str] = []
        images: list[list[Image.Image] | None] = []
        media_metadata: list[dict[str, Any] | None] = []
        for item in batch:
            text, sample_images, sample_media_metadata = extract_retrieval_payload(
                item,
                wrapper_name=type(self).__name__,
            )
            texts.append(text)
            images.append(sample_images)
            media_metadata.append(sample_media_metadata)
        return self(texts, images, media_metadata=media_metadata)

    def _process_single(
        self,
        text: str,
        images: list[Image.Image] | None,
        media_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process one finalized sample according to Qwen3-VL input rules."""
        text = self._normalize_visual_text(text)
        has_image_token = self.IMAGE_TOKEN in text
        has_video_token = self.VIDEO_TOKEN in text
        has_visual_token = has_image_token or has_video_token
        media = images or []
        if any(img is None for img in media):
            raise ValueError("Qwen3VLProcessorWrapper received None inside images list")

        if not media:
            if has_visual_token:
                raise ValueError("Visual tokens present in text but no media provided")
            processor = cast(Any, self.processor)
            return processor(
                text=[text or " "],
                images=None,
                return_tensors="pt",
                return_mm_token_type_ids=True,
                max_length=self.max_length,
                truncation=bool(self.max_length),
            )

        if not has_visual_token:
            raise ValueError("Media provided but text contains no visual token")
        if has_image_token and has_video_token:
            raise ValueError("Ambiguous Qwen visual sample: both image and video tokens present")

        if has_image_token:
            processor = cast(Any, self.processor)
            return processor(
                text=[text],
                images=media,
                return_tensors="pt",
                return_mm_token_type_ids=True,
                images_kwargs={"input_data_format": self._input_data_format},
            )

        # Video mode
        # Qwen3VL video_processor only uses size dict, not min_pixels/max_pixels
        kwargs = {}
        if self._video_size is not None:
            kwargs["videos_kwargs"] = {"size": self._video_size}
        # Keep the public Qwen3-VL runtime behavior while allowing dataset
        # metadata to override the default placeholder timing fields. Qwen3VL uses frame
        # indices plus fps to build timestamped placeholders, so we preserve
        # sampled frame indices whenever datasets provide them.
        timing_metadata = {
            "fps": 1.0,
            "total_num_frames": len(media),
            "frames_indices": list(range(len(media))),
        }
        if media_metadata:
            if isinstance(media_metadata.get("fps"), (int, float)):
                timing_metadata["fps"] = float(media_metadata["fps"])
            if isinstance(media_metadata.get("total_num_frames"), int):
                timing_metadata["total_num_frames"] = int(media_metadata["total_num_frames"])
            frame_indices = media_metadata.get("frames_indices")
            if not isinstance(frame_indices, list):
                frame_indices = media_metadata.get("sampled_indices")
            if (
                isinstance(frame_indices, list)
                and frame_indices
                and all(isinstance(idx, int) for idx in frame_indices)
            ):
                timing_metadata["frames_indices"] = [int(idx) for idx in frame_indices]
        media = self._ensure_video_temporal_factor(media, timing_metadata)
        kwargs["video_metadata"] = [timing_metadata]
        processor = cast(Any, self.processor)
        return processor(
            text=[text],
            videos=[media],
            return_tensors="pt",
            return_mm_token_type_ids=True,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation."""
        img_proc = cast(Any, self.processor).image_processor
        vid_proc = getattr(self.processor, "video_processor", None)
        parts = [
            f"max_length={self.max_length}",
            f"image_min_pixels={img_proc.min_pixels}",
            f"image_max_pixels={img_proc.max_pixels}",
        ]
        if vid_proc is not None:
            vid_size = vid_proc.size
            parts.append(f"video_min_pixels={vid_size.get('shortest_edge')}")
            parts.append(f"video_max_pixels={vid_size.get('longest_edge')}")
        return f"Qwen3VLProcessorWrapper({', '.join(parts)})"
