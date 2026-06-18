"""VATEX training dataset runtime."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
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
DEFAULT_QUERY_INSTRUCTION = "Find a video that matches the following caption:"
DEFAULT_POSITIVE_INSTRUCTION = "Understand the content of the provided video."
DATASET_NAME = "VATEX"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
CAPTION_SELECTION_VALUES = {"first", "random"}
VATEX_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline", "caption_selection", "caption_field"},
    "positive": {"instruction", "trailing_newline"},
    "negative": {"trailing_newline"},
}
CAPTION_FIELD_VALUES = {"en_caption", "ch_caption"}


@dataclass(frozen=True)
class VatexTrainTransformConfig:
    """Default transform configuration for VATEX training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_trailing_newline: str = "ensure_single"
    query_caption_selection: str = "first"
    query_caption_field: str = "en_caption"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
    positive_trailing_newline: str = "ensure_single"
    negative_trailing_newline: str = "ensure_single"


def _apply_trailing_newline(text: str, mode: str) -> str:
    """Apply one side's trailing-newline policy."""

    if mode not in TRAILING_NEWLINE_VALUES:
        raise ValueError(f"{DATASET_NAME} trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}.")
    if mode == "preserve":
        return text
    if mode == "strip":
        return text.rstrip("\n")
    return text.rstrip("\n") + "\n" if text else ""


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> VatexTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name=DATASET_NAME,
        allowed_keys=VATEX_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"{DATASET_NAME} transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    query_caption_selection = query.get("caption_selection", "first")
    if query_caption_selection not in CAPTION_SELECTION_VALUES:
        raise ValueError(
            f"{DATASET_NAME} transform.query.caption_selection must be one of {sorted(CAPTION_SELECTION_VALUES)}."
        )
    query_caption_field = query.get("caption_field", "en_caption")
    if query_caption_field not in CAPTION_FIELD_VALUES:
        raise ValueError(f"{DATASET_NAME} transform.query.caption_field must be one of {sorted(CAPTION_FIELD_VALUES)}.")
    return VatexTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name=DATASET_NAME,
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name=DATASET_NAME,
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        query_caption_selection=query_caption_selection,
        query_caption_field=query_caption_field,
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name=DATASET_NAME,
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _extension_key(record: Mapping[str, Any]) -> str:
    """Resolve the VATEX frame-unit key."""

    return str(record["video"])


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
    raise ValueError(f"{DATASET_NAME} caption_selection must be one of {sorted(CAPTION_SELECTION_VALUES)}.")


def build_vatex_query_text(
    caption: Any,
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
    caption_selection: str = "first",
) -> str:
    """Build the VATEX text-to-video query text."""

    body = normalize_text_whitespace(_select_caption(caption, selection=caption_selection))
    text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _apply_trailing_newline(text, trailing_newline)


def build_vatex_positive_text(
    *,
    instruction: str = DEFAULT_POSITIVE_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the VATEX positive video text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline)


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline),
        "media": [],
    }


def _build_video_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample the joined pre-extracted full-video frame-unit."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if record.get("video_id") is not None:
        media_metadata["video_id"] = str(record["video_id"])
    if record.get("video") is not None:
        media_metadata["video"] = str(record["video"])
    if not images:
        raise ValueError("VATEX row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_vatex_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: VatexTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined VATEX row into a TrainSample."""

    return {
        "query": {
            "text": build_vatex_query_text(
                record.get(config.query_caption_field),
                instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
                caption_selection=config.query_caption_selection,
            ),
            "media": [],
        },
        "positive": {
            "text": build_vatex_positive_text(
                instruction=config.positive_instruction,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": _build_video_media(record, num_frames=num_frames),
        },
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "video_id": record.get("video_id"),
            "video": record.get("video"),
            "caption_field": config.query_caption_field,
            "en_caption": record.get("en_caption"),
            "ch_caption": record.get("ch_caption"),
        },
    }


def build_vatex_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default VATEX transform."""

    return partial(
        transform_vatex_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("vatex_train")
class VatexTrainDataset(SampleLanceDataset):
    """VATEX bilingual text-to-video retrieval training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        num_frames: int = 8,
        dataset_name: str = DATASET_NAME,
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("VatexTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for VatexTrainDataset.")
        root = path
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_vatex_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=f"{root}/data/{split}.lance",
            read_columns=requested_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=f"{root}/data/frames.lance",
                    key=_extension_key,
                    column_name="frame_unit_key",
                    read_columns=[
                        "frame_unit_type",
                        "source_key",
                        "source_path",
                        "frames",
                        "sampled_frame_count",
                        "source_total_num_frames",
                        "clip_total_num_frames",
                        "fps",
                        "duration",
                        "clip_start",
                        "clip_end",
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
        self.dataset_name = dataset_name
        self.split = split
        self.num_frames = num_frames
        self.subset = None
        self.variant = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VatexTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("VATEX config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("VATEX config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_POSITIVE_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "VatexTrainDataset",
    "VatexTrainTransformConfig",
    "build_vatex_positive_text",
    "build_vatex_query_text",
    "build_vatex_train_default_transform",
    "transform_vatex_train_sample",
]
