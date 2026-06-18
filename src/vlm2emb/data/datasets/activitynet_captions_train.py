"""ActivityNet Captions training dataset runtime."""

from __future__ import annotations

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

SUPPORTED_TASK_ARCHETYPES = ("video_segment_retrieval",)
DEFAULT_QUERY_INSTRUCTION = "Find the video segment that corresponds to the following description:"
DEFAULT_POSITIVE_INSTRUCTION = "Understand the content of the provided video segment."
DATASET_NAME = "ActivityNet-Captions"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
ACTIVITYNET_CAPTIONS_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"instruction", "trailing_newline"},
    "negative": {"trailing_newline"},
}


@dataclass(frozen=True)
class ActivitynetCaptionsTransformConfig:
    """Default transform configuration for ActivityNet Captions rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_trailing_newline: str = "ensure_single"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> ActivitynetCaptionsTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name=DATASET_NAME,
        allowed_keys=ACTIVITYNET_CAPTIONS_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"{DATASET_NAME} transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    return ActivitynetCaptionsTransformConfig(
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


def build_activitynet_captions_query_text(
    sentence: str,
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the ActivityNet Captions segment retrieval query text."""

    body = normalize_text_whitespace(sentence)
    text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _apply_trailing_newline(text, trailing_newline, dataset_name=DATASET_NAME)


def build_activitynet_captions_positive_text(
    *,
    instruction: str = DEFAULT_POSITIVE_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the ActivityNet Captions positive video-segment text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name=DATASET_NAME)


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name=DATASET_NAME),
        "media": [],
    }


def _build_positive_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample the pre-extracted frame-unit for the labeled segment."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if record.get("video") is not None:
        media_metadata["video"] = str(record["video"])
    if record.get("video_id") is not None:
        media_metadata["video_id"] = str(record["video_id"])
    if not images:
        raise ValueError("ActivityNet Captions row resolved to zero sampled positive clip frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_activitynet_captions_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: ActivitynetCaptionsTransformConfig,
) -> dict[str, Any]:
    """Transform one joined ActivityNet Captions row into a TrainSample."""

    return {
        "query": {
            "text": build_activitynet_captions_query_text(
                str(record.get("segment_sentence", "") or ""),
                instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": [],
        },
        "positive": {
            "text": build_activitynet_captions_positive_text(
                instruction=config.positive_instruction,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": _build_positive_media(record, num_frames=num_frames),
        },
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "video_id": record.get("video_id"),
            "video": record.get("video"),
            "source": record.get("source"),
            "duration": record.get("duration"),
            "segment_index": record.get("segment_index"),
            "segment_start": record.get("segment_start"),
            "segment_end": record.get("segment_end"),
            "caption": record.get("caption"),
        },
    }


def build_activitynet_captions_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default ActivityNet Captions transform."""

    return partial(
        transform_activitynet_captions_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("activitynet_captions_train")
class ActivitynetCaptionsTrainDataset(SampleLanceDataset):
    """ActivityNet Captions text-to-video-segment training dataset."""

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
            raise TypeError("ActivitynetCaptionsTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for ActivitynetCaptionsTrainDataset.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_activitynet_captions_default_transform(
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(root / "data" / f"{split}.lance"),
            read_columns=requested_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "frames.lance"),
                    key="frame_unit_key",
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
    def from_config(cls, config: Mapping[str, Any]) -> ActivitynetCaptionsTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("ActivityNet Captions config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("ActivityNet Captions config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_POSITIVE_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "ActivitynetCaptionsTrainDataset",
    "ActivitynetCaptionsTransformConfig",
    "build_activitynet_captions_default_transform",
    "build_activitynet_captions_positive_text",
    "build_activitynet_captions_query_text",
    "transform_activitynet_captions_train_sample",
]
