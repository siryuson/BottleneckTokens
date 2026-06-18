"""Breakfast training dataset runtime."""

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
from vlm2emb.data.datasets.transforms import normalize_side_transform_mapping
from vlm2emb.data.datasets.video_frame_units import sample_preextracted_frame_unit

SUPPORTED_TASK_ARCHETYPES = ("classification",)
DEFAULT_QUERY_INSTRUCTION = "Recognize the breakfast type that the person is cooking in the video."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
BREAKFAST_TRANSFORM_KEYS = {
    "query": {"instruction", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class BreakfastTransformConfig:
    """Default transform configuration for Breakfast training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_visual_token_placement: str = "own_line"
    query_trailing_newline: str = "ensure_single"
    positive_trailing_newline: str = "strip"
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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> BreakfastTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="Breakfast",
        allowed_keys=BREAKFAST_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    visual_token_placement = query.get("visual_token_placement", "own_line")
    if visual_token_placement != "own_line":
        raise ValueError("Breakfast transform.query.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("Breakfast transform.negative.empty must be 'empty_multimodal_input'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"Breakfast transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    instruction = query.get("instruction", DEFAULT_QUERY_INSTRUCTION)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Breakfast transform.query.instruction must be a non-empty string.")
    return BreakfastTransformConfig(
        query_instruction=instruction,
        query_visual_token_placement=visual_token_placement,
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def build_breakfast_query_text(
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the Breakfast video classification query text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="Breakfast")


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="Breakfast"),
        "media": [],
    }


def _positive_label(record: Mapping[str, Any]) -> str:
    """Resolve the canonical training label while preserving raw_label as fallback."""

    return str(record.get("label") or record.get("raw_label") or "")


def _build_video_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample the joined pre-extracted full-video frame-unit."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if not images:
        raise ValueError("Breakfast row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_breakfast_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: BreakfastTransformConfig,
) -> dict[str, Any]:
    """Transform one joined Breakfast row into a TrainSample."""

    metadata = {
        key: value
        for key, value in {
            "dataset_name": dataset_name,
            "split": split,
            "video_id": record.get("video_id"),
            "video_path": record.get("video_path"),
            "label": record.get("label"),
            "raw_label": record.get("raw_label"),
            "participant": record.get("participant"),
            "camera": record.get("camera"),
            "source_split": record.get("source_split"),
            "labels": record.get("labels"),
        }.items()
        if value is not None
    }
    return {
        "query": {
            "text": build_breakfast_query_text(
                instruction=config.query_instruction,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_video_media(record, num_frames=num_frames),
        },
        "positive": _text_side(_positive_label(record), config.positive_trailing_newline),
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_breakfast_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default Breakfast transform."""

    return partial(
        transform_breakfast_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


def _split_path(root: Path, split: str) -> Path:
    """Resolve one converted Breakfast split table path."""

    if split == "train":
        split = "archive_train_non_cam01"
    return root / "data" / f"{split}.lance"


@AutoDataset.register("breakfast_train")
class BreakfastTrainDataset(SampleLanceDataset):
    """Breakfast training dataset over raw split rows and joined video bytes."""

    def __init__(
        self,
        path: str,
        split: str = "archive_train_non_cam01",
        num_frames: int = 8,
        dataset_name: str = "Breakfast",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("BreakfastTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_breakfast_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(_split_path(root, split)),
            read_columns=requested_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "frames.lance"),
                    key="video_path",
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
        self.split = split
        self.num_frames = num_frames
        self.dataset_name = dataset_name

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> BreakfastTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("Breakfast config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("Breakfast config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "BreakfastTrainDataset",
    "build_breakfast_query_text",
    "build_breakfast_train_default_transform",
    "transform_breakfast_train_sample",
]
