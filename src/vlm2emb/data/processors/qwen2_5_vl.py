"""Qwen2.5-VL series processor wrapper.

Qwen2.5-VL 
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

logger = logging.getLogger(__name__)


@AutoProcessorWrapper.register("qwen2_5_vl")
class Qwen2_5VLProcessorWrapper:
    """ProcessorWrapper for Qwen2.5-VL models."""

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
        fps: float = 2.0,
        wrap_visual_tokens_with_boundaries: bool = True,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.fps = fps
        self.wrap_visual_tokens_with_boundaries = wrap_visual_tokens_with_boundaries
        self._input_data_format = self._detect_input_data_format()
        self._video_size = self._get_video_size()

    @property
    def tokenizer(self):
        return cast(Any, self.processor).tokenizer

    @property
    def vision_token_ids(self) -> list[int]:
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

        Keep boundary wrapping enabled by default so Qwen2.5-VL uses the same
        production-safe normalization policy as the Qwen2-VL wrapper. The
        optional switch exists for controlled regression experiments only.
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
        try:
            from transformers.image_processing_utils_fast import BaseImageProcessorFast

            is_fast = isinstance(cast(Any, self.processor).image_processor, BaseImageProcessorFast)
        except ImportError:
            is_fast = False
        return ChannelDimension.FIRST if is_fast else ChannelDimension.LAST

    def _get_video_size(self) -> dict[str, int] | None:
        vid_proc = getattr(self.processor, "video_processor", None)
        if vid_proc is None:
            return None
        img_proc = cast(Any, self.processor).image_processor
        if vid_proc.max_pixels != img_proc.max_pixels or vid_proc.min_pixels != img_proc.min_pixels:
            return {"shortest_edge": vid_proc.min_pixels, "longest_edge": vid_proc.max_pixels}
        return None

    def save_pretrained(self, *args, **kwargs):
        return self.processor.save_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Qwen2_5VLProcessorWrapper:
        config = config.copy()
        config.pop("type", None)

        processor_name = config.pop("processor_name", None)
        processor = config.pop("processor", None)
        fps = config.pop("fps", 2.0)

        old_min = config.pop("min_pixels", 28 * 28 * 4)
        old_max = config.pop("max_pixels", 28 * 28 * 1280)
        image_min_pixels = config.pop("image_min_pixels", old_min)
        image_max_pixels = config.pop("image_max_pixels", old_max)
        video_min_pixels = config.pop("video_min_pixels", image_min_pixels)
        video_max_pixels = config.pop("video_max_pixels", 28 * 28 * 768)

        if processor is None:
            if processor_name is None:
                raise ValueError("Either 'processor' or 'processor_name' required in config")
            processor = AutoProcessor.from_pretrained(
                processor_name,
                trust_remote_code=True,
                min_pixels=image_min_pixels,
                max_pixels=image_max_pixels,
                size={"shortest_edge": image_min_pixels, "longest_edge": image_max_pixels},
            )

        vid_proc = getattr(processor, "video_processor", None)
        if vid_proc is not None and (
            video_min_pixels != image_min_pixels or video_max_pixels != image_max_pixels
        ):
            vid_proc.min_pixels = video_min_pixels
            vid_proc.max_pixels = video_max_pixels
            vid_proc.size = {
                "shortest_edge": video_min_pixels,
                "longest_edge": video_max_pixels,
            }

        return cls(processor=processor, fps=fps, **config)

    def __call__(
        self,
        texts: list[str],
        images: list[list[Image.Image] | None],
        media_metadata: list[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, torch.Tensor | None]]:
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
                "second_per_grid_ts": result.get("second_per_grid_ts"),
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
        text = self._normalize_visual_text(text)
        has_image_token = self.IMAGE_TOKEN in text
        has_video_token = self.VIDEO_TOKEN in text
        has_visual_token = has_image_token or has_video_token
        media = images or []
        if any(img is None for img in media):
            raise ValueError("Qwen2_5VLProcessorWrapper received None inside images list")

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

        kwargs: dict[str, Any] = {
            "videos_kwargs": {"fps": self.fps},
        }
        if media_metadata and isinstance(media_metadata.get("fps"), (int, float)):
            kwargs["videos_kwargs"]["fps"] = float(media_metadata["fps"])
        if self._video_size is not None:
            kwargs["videos_kwargs"].update(
                {
                    "size": self._video_size,
                    "min_pixels": self._video_size["shortest_edge"],
                    "max_pixels": self._video_size["longest_edge"],
                }
            )
        processor = cast(Any, self.processor)
        return processor(
            text=[text],
            videos=[media],
            return_tensors="pt",
            return_mm_token_type_ids=True,
            **kwargs,
        )

    def __repr__(self) -> str:
        img_proc = cast(Any, self.processor).image_processor
        vid_proc = getattr(self.processor, "video_processor", None)
        parts = [
            f"max_length={self.max_length}",
            f"fps={self.fps}",
            f"image_min_pixels={img_proc.min_pixels}",
            f"image_max_pixels={img_proc.max_pixels}",
        ]
        if vid_proc is not None:
            parts.append(f"video_min_pixels={vid_proc.min_pixels}")
            parts.append(f"video_max_pixels={vid_proc.max_pixels}")
        return f"Qwen2_5VLProcessorWrapper({', '.join(parts)})"
