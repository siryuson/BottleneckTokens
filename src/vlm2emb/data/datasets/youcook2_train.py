"""YouCook2 training dataset runtime."""

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
from vlm2emb.data.utils.video_paths import normalize_youcook2_video_path

SUPPORTED_TASK_ARCHETYPES = ("video_retrieval",)
DEFAULT_QUERY_INSTRUCTION = "Find a video that demonstrates the following action while making a recipe:"
DEFAULT_POSITIVE_INSTRUCTION = "Understand the content of the provided video."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
YOUCOOK2_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"instruction", "trailing_newline"},
    "negative": {"trailing_newline"},
}


@dataclass(frozen=True)
class Youcook2TrainTransformConfig:
    """Default transform configuration for YouCook2 training rows."""

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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> Youcook2TrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="YouCook2",
        allowed_keys=YOUCOOK2_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"YouCook2 transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    return Youcook2TrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="YouCook2",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="YouCook2",
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="YouCook2",
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _extension_key(record: Mapping[str, Any]) -> str:
    """Resolve the normalized YouCook2 video table key."""

    return normalize_youcook2_video_path(str(record["video_path"]))


def build_youcook2_query_text(
    caption: str,
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the YouCook2 text-to-video query text."""

    body = normalize_text_whitespace(caption)
    text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _apply_trailing_newline(text, trailing_newline, dataset_name="YouCook2")


def build_youcook2_positive_text(
    *,
    instruction: str = DEFAULT_POSITIVE_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the YouCook2 positive video text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="YouCook2")


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="YouCook2"),
        "media": [],
    }


def _build_video_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Build media from pre-extracted row-level frames."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if record.get("video_path") is not None:
        media_metadata["source_video_path"] = str(record["video_path"])
        media_metadata["video_path"] = normalize_youcook2_video_path(str(record["video_path"]))
    if record.get("video_id") is not None:
        media_metadata["video_id"] = str(record["video_id"])
    if not images:
        raise ValueError("YouCook2 row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_youcook2_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: Youcook2TrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined YouCook2 row into a TrainSample."""

    normalized_video_path = normalize_youcook2_video_path(str(record["video_path"]))
    return {
        "query": {
            "text": build_youcook2_query_text(
                str(record.get("caption", "") or ""),
                instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": [],
        },
        "positive": {
            "text": build_youcook2_positive_text(
                instruction=config.positive_instruction,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": _build_video_media(record, num_frames=num_frames),
        },
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "video_path": record.get("video_path"),
            "normalized_video_path": normalized_video_path,
            "video_id": Path(normalized_video_path).stem,
            "recipe_type": record.get("recipe_type"),
            "id": record.get("id"),
            "num_frames": record.get("num_frames"),
            "height": record.get("height"),
            "width": record.get("width"),
            "channels": record.get("channels"),
        },
    }


def build_youcook2_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default YouCook2 transform."""

    return partial(
        transform_youcook2_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("youcook2_train")
class Youcook2TrainDataset(SampleLanceDataset):
    """YouCook2 text-to-video recipe action retrieval training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        num_frames: int = 8,
        dataset_name: str = "YouCook2",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("Youcook2TrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for Youcook2TrainDataset.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_youcook2_train_default_transform(
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
                        "frames",
                        "frame_unit_type",
                        "source_key",
                        "source_path",
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
    def from_config(cls, config: Mapping[str, Any]) -> Youcook2TrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("YouCook2 config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("YouCook2 config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_POSITIVE_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "Youcook2TrainDataset",
    "Youcook2TrainTransformConfig",
    "build_youcook2_positive_text",
    "build_youcook2_query_text",
    "build_youcook2_train_default_transform",
    "transform_youcook2_train_sample",
]
