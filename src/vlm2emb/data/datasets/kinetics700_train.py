"""Kinetics-700 training dataset runtime."""

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
from vlm2emb.data.datasets.transforms import normalize_side_transform_mapping
from vlm2emb.data.datasets.video_frame_units import sample_preextracted_frame_unit

SUPPORTED_TASK_ARCHETYPES = ("classification",)
DEFAULT_QUERY_INSTRUCTION = "Recognize the category of the video content."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
KINETICS700_TRANSFORM_KEYS = {
    "query": {"instruction", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class Kinetics700TransformConfig:
    """Default transform configuration for Kinetics-700 training rows."""

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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> Kinetics700TransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="Kinetics-700",
        allowed_keys=KINETICS700_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    visual_token_placement = query.get("visual_token_placement", "own_line")
    if visual_token_placement != "own_line":
        raise ValueError("Kinetics-700 transform.query.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("Kinetics-700 transform.negative.empty must be 'empty_multimodal_input'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"Kinetics-700 transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    instruction = query.get("instruction", DEFAULT_QUERY_INSTRUCTION)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Kinetics-700 transform.query.instruction must be a non-empty string.")
    return Kinetics700TransformConfig(
        query_instruction=instruction,
        query_visual_token_placement=visual_token_placement,
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def build_kinetics700_query_text(
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the Kinetics-700 video classification query text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="Kinetics-700")


def _label_text(record: Mapping[str, Any]) -> str:
    """Read the positive label from raw Kinetics-700 rows."""

    for key in ("label", "pos_text", "label_text", "class_name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="Kinetics-700"),
        "media": [],
    }


def _build_video_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample the joined pre-extracted full-video frame-unit."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if not images:
        raise ValueError("Kinetics-700 row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_kinetics700_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: Kinetics700TransformConfig,
) -> dict[str, Any]:
    """Transform one joined Kinetics-700 row into a TrainSample."""

    metadata = {
        key: value
        for key, value in {
            "dataset_name": dataset_name,
            "split": split,
            "video_id": record.get("video_id"),
            "video_path": record.get("video_path"),
            "label": record.get("label"),
            "youtube_id": record.get("youtube_id"),
            "time_start": record.get("time_start"),
            "time_end": record.get("time_end"),
            "source_split": record.get("split"),
        }.items()
        if value is not None
    }
    return {
        "query": {
            "text": build_kinetics700_query_text(
                instruction=config.query_instruction,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_video_media(record, num_frames=num_frames),
        },
        "positive": _text_side(_label_text(record), config.positive_trailing_newline),
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_kinetics700_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default Kinetics-700 transform."""

    return partial(
        transform_kinetics700_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


def _split_path(root: Path, split: str) -> Path:
    """Resolve one converted Kinetics-700 split table path."""

    if split == "train":
        split = "train_without_mmeb_v2_eval"
    return root / "data" / f"{split}.lance"


def _build_sample_index(*, base_length: int, max_samples: int | None, sample_seed: int | None) -> list[int] | None:
    """Build an optional deterministic index map for config-level sample caps."""

    if max_samples is None:
        return None
    if max_samples <= 0:
        raise ValueError("Kinetics-700 max_samples must be a positive integer when set.")
    capped_length = min(max_samples, base_length)
    if sample_seed is None:
        return list(range(capped_length))
    indices = list(range(base_length))
    random.Random(sample_seed).shuffle(indices)
    return indices[:capped_length]


@AutoDataset.register("kinetics700_train")
class Kinetics700TrainDataset(SampleLanceDataset):
    """Kinetics-700 training dataset over raw split rows and joined video bytes."""

    def __init__(
        self,
        path: str,
        split: str = "train_without_mmeb_v2_eval",
        num_frames: int = 32,
        dataset_name: str = "Kinetics-700",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        max_samples: int | None = None,
        sample_seed: int | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("Kinetics700TrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        lance_path = _split_path(root, split)
        extensions = [
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
        ]
        effective_transform = (
            transform
            if transform is not None
            else build_kinetics700_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(lance_path),
            read_columns=requested_columns,
            extensions=extensions,
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.split = split
        self.num_frames = num_frames
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.sample_seed = sample_seed
        self._sample_index: list[int] | None = None

    def _effective_index(self, idx: int) -> int:
        """Map public sample indices through the optional deterministic cap."""

        sample_index = self._sample_index
        if sample_index is None:
            return idx
        return sample_index[idx]

    def __len__(self) -> int:
        base_length = super().__len__()
        if self.max_samples is None:
            return base_length
        if self._sample_index is None:
            self._sample_index = _build_sample_index(
                base_length=base_length,
                max_samples=self.max_samples,
                sample_seed=self.sample_seed,
            )
        return len(self._sample_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._transform_record(self._read_records([self._effective_index(idx)])[0])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        length = len(self)
        for idx in indices:
            if idx < 0 or idx >= length:
                raise IndexError(f"Index {idx} out of range [0, {length})")
        return super().__getitems__([self._effective_index(idx) for idx in indices])

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Kinetics700TrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("Kinetics-700 config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("Kinetics-700 config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "Kinetics700TrainDataset",
    "build_kinetics700_query_text",
    "build_kinetics700_train_default_transform",
    "transform_kinetics700_train_sample",
]
