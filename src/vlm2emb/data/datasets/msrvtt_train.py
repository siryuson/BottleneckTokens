"""MSR-VTT training dataset runtime."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import LanceTableExtension, SampleLanceDataset, SampleTransform
from vlm2emb.data.datasets.const import (
    LEGACY_VIDEO_TOKENS,
    STANDARD_VIDEO_TOKEN,
    canonicalize_multimodal_text,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    join_instruction_body,
    normalize_instruction_body_separator,
    normalize_instruction_text,
    normalize_side_transform_mapping,
)
from vlm2emb.data.datasets.video_frame_units import sample_preextracted_frame_unit

SUPPORTED_TASK_ARCHETYPES = ("video_retrieval",)
DEFAULT_QUERY_INSTRUCTION = "Find a video that contains the following visual content:"
DEFAULT_POSITIVE_INSTRUCTION = "Understand the content of the provided video."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
CAPTION_SELECTION_VALUES = {"first", "random"}
MSRVTT_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline", "caption_selection"},
    "positive": {"instruction", "visual_token_placement", "trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class MsrvttTransformConfig:
    """Default transform configuration for MSR-VTT training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_trailing_newline: str = "ensure_single"
    query_caption_selection: str = "first"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
    positive_visual_token_placement: str = "own_line"
    positive_trailing_newline: str = "ensure_single"
    negative_trailing_newline: str = "ensure_single"


def _apply_trailing_newline(text: str, mode: str, *, dataset_name: str) -> str:
    """Apply one side's trailing-newline policy."""

    if mode not in TRAILING_NEWLINE_VALUES:
        raise ValueError(f"{dataset_name} trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}.")
    if mode == "preserve":
        return text
    if mode == "strip":
        return text.rstrip("\n")
    return text.rstrip("\n") + "\n" if text else ""


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> MsrvttTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="MSR-VTT",
        allowed_keys=MSRVTT_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    positive_visual_token_placement = positive.get("visual_token_placement", "own_line")
    if positive_visual_token_placement != "own_line":
        raise ValueError("MSR-VTT transform.positive.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("MSR-VTT transform.negative.empty must be 'empty_multimodal_input'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"MSR-VTT transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    query_caption_selection = query.get("caption_selection", "first")
    if query_caption_selection not in CAPTION_SELECTION_VALUES:
        raise ValueError(
            f"MSR-VTT transform.query.caption_selection must be one of {sorted(CAPTION_SELECTION_VALUES)}."
        )
    return MsrvttTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="MSR-VTT",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="MSR-VTT",
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        query_caption_selection=query_caption_selection,
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="MSR-VTT",
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_visual_token_placement=positive_visual_token_placement,
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _select_caption(value: Any, *, selection: str) -> str:
    """Return one caption from source rows according to the configured policy."""

    if isinstance(value, list):
        captions = [str(caption) for caption in value if caption is not None and str(caption).strip()]
    else:
        captions = [str(value)] if value is not None and str(value).strip() else []
    if not captions:
        return ""
    if selection == "first":
        return captions[0]
    if selection == "random":
        return random.choice(captions)
    raise ValueError(f"MSR-VTT caption_selection must be one of {sorted(CAPTION_SELECTION_VALUES)}.")


def build_msrvtt_query_text(
    caption: Any,
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
    caption_selection: str = "first",
) -> str:
    """Build the MSR-VTT text-to-video query text."""

    body = normalize_text_whitespace(_select_caption(caption, selection=caption_selection))
    text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _apply_trailing_newline(text, trailing_newline, dataset_name="MSR-VTT")


def build_msrvtt_positive_text(
    *,
    instruction: str = DEFAULT_POSITIVE_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the MSR-VTT positive video text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="MSR-VTT")


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="MSR-VTT"),
        "media": [],
    }


def _build_video_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Build media from pre-extracted frame-unit bytes."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if record.get("video_id") is not None:
        media_metadata["video_id"] = str(record["video_id"])
    if record.get("video") is not None:
        media_metadata["video"] = str(record["video"])
    if not images:
        raise ValueError("MSR-VTT row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_msrvtt_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: MsrvttTransformConfig,
) -> dict[str, Any]:
    """Transform one joined MSR-VTT row into a TrainSample."""

    metadata = {
        key: value
        for key, value in {
            "dataset_name": dataset_name,
            "split": split,
            "video_id": record.get("video_id"),
            "video": record.get("video"),
            "source": record.get("source"),
            "category": record.get("category"),
            "url": record.get("url"),
            "start_time": record.get("start time"),
            "end_time": record.get("end time"),
            "id": record.get("id"),
        }.items()
        if value is not None
    }
    return {
        "query": {
            "text": build_msrvtt_query_text(
                record.get("caption"),
                instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
                caption_selection=config.query_caption_selection,
            ),
            "media": [],
        },
        "positive": {
            "text": build_msrvtt_positive_text(
                instruction=config.positive_instruction,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": _build_video_media(record, num_frames=num_frames),
        },
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_msrvtt_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default MSR-VTT transform."""

    return partial(
        transform_msrvtt_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("msrvtt_train")
class MsrvttTrainDataset(SampleLanceDataset):
    """MSR-VTT text-to-video training dataset over raw split rows."""

    def __init__(
        self,
        path: str,
        split: str = "train_9k_without_mmeb_v2_eval",
        num_frames: int = 8,
        dataset_name: str = "MSR-VTT",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("MsrvttTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns.")
        root = Path(path)
        lance_path = root / "data" / f"{split}.lance"
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_msrvtt_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(lance_path),
            read_columns=requested_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "frames.lance"),
                    key="video",
                    column_name="frame_unit_key",
                    read_columns=[
                        "frames",
                        "frame_unit_type",
                        "source_key",
                        "source_path",
                        "sampled_frame_count",
                        "source_total_num_frames",
                        "clip_total_num_frames",
                        "fps",
                        "duration",
                        "sampled_indices",
                        "sampled_timestamps",
                        "decode_backend",
                    ],
                )
            ],
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.split = split
        self.num_frames = num_frames
        self.dataset_name = dataset_name

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MsrvttTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("MSR-VTT config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("MSR-VTT config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_POSITIVE_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "MsrvttTrainDataset",
    "build_msrvtt_positive_text",
    "build_msrvtt_query_text",
    "build_msrvtt_train_default_transform",
    "transform_msrvtt_train_sample",
]
