"""HMDB51 training dataset runtime."""

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
from vlm2emb.data.utils.video import sample_video_bytes

SUPPORTED_TASK_ARCHETYPES = ("classification",)
DEFAULT_QUERY_INSTRUCTION = "What actions or objects interactions are the person in the video doing?"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
HMDB51_TRANSFORM_KEYS = {
    "query": {"instruction", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class Hmdb51TransformConfig:
    """Default transform configuration for HMDB51 training rows."""

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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> Hmdb51TransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="HMDB51",
        allowed_keys=HMDB51_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    visual_token_placement = query.get("visual_token_placement", "own_line")
    if visual_token_placement != "own_line":
        raise ValueError("HMDB51 transform.query.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("HMDB51 transform.negative.empty must be 'empty_multimodal_input'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"HMDB51 transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    instruction = query.get("instruction", DEFAULT_QUERY_INSTRUCTION)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("HMDB51 transform.query.instruction must be a non-empty string.")
    return Hmdb51TransformConfig(
        query_instruction=instruction,
        query_visual_token_placement=visual_token_placement,
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def build_hmdb51_query_text(
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the HMDB51 video classification query text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {instruction}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="HMDB51")


def _label_text(record: Mapping[str, Any]) -> str:
    """Read the canonical positive label from raw or legacy HMDB51 rows."""

    for key in ("label", "pos_text", "label_text", "class_name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _negative_text(record: Mapping[str, Any]) -> str:
    """Read the optional negative label from raw or legacy HMDB51 rows."""

    value = record.get("negative_text")
    return "" if value is None else str(value)


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="HMDB51"),
        "media": [],
    }


def _build_video_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read pre-extracted frames, falling back to legacy raw video bytes."""

    if isinstance(record.get("frames"), list):
        images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    else:
        video = record.get("video")
        if video is None:
            video = record.get("video_bytes")
        if not isinstance(video, (bytes, bytearray)):
            raise TypeError("HMDB51 row is missing required video frames or raw video bytes.")
        images, media_metadata = sample_video_bytes(
            bytes(video),
            num_frames=num_frames,
            source_path=record.get("video_path") or record.get("video"),
        )
        fps = record.get("fps")
        total_num_frames = record.get("total_num_frames")
        if isinstance(fps, (int, float)) and fps > 0:
            media_metadata["fps"] = float(fps)
        if isinstance(total_num_frames, int) and total_num_frames >= 0:
            media_metadata["total_num_frames"] = total_num_frames
    if not images:
        raise ValueError("HMDB51 row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_hmdb51_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: Hmdb51TransformConfig,
) -> dict[str, Any]:
    """Transform one joined HMDB51 row into a TrainSample."""

    metadata = {
        key: value
        for key, value in {
            "dataset_name": dataset_name,
            "split": split,
            "video_id": record.get("video_id"),
            "video_path": record.get("video_path"),
            "split_tag": record.get("split_tag"),
            "split_file": record.get("split_file"),
        }.items()
        if value is not None
    }
    return {
        "query": {
            "text": build_hmdb51_query_text(
                instruction=config.query_instruction,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_video_media(record, num_frames=num_frames),
        },
        "positive": _text_side(_label_text(record), config.positive_trailing_newline),
        "negative": _text_side(_negative_text(record), config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_hmdb51_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default HMDB51 transform."""

    return partial(
        transform_hmdb51_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


def _raw_split_path(root: Path, split: str) -> Path:
    """Resolve the preferred raw-preserving split table path."""

    if split == "train":
        return root / "data" / "vlm2vec_train.lance"
    return root / "data" / f"{split}.lance"


def _legacy_split_path(root: Path, split: str) -> Path:
    """Resolve the legacy split-membership table path for existing roots."""

    if split == "train":
        split = "vlm2vec_train"
    return root / "data" / "splits" / f"{split}.lance"


@AutoDataset.register("hmdb51_train")
class Hmdb51TrainDataset(SampleLanceDataset):
    """HMDB51 training dataset over raw split rows and joined video bytes."""

    def __init__(
        self,
        path: str,
        split: str = "vlm2vec_train",
        num_frames: int = 8,
        dataset_name: str = "HMDB51",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        allow_raw_video: bool = False,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("Hmdb51TrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        raw_path = _raw_split_path(root, split)
        legacy_path = _legacy_split_path(root, split)
        frame_units_path = root / "data" / "frames.lance"
        use_legacy_layout = not frame_units_path.exists() and not raw_path.exists() and legacy_path.exists()
        if use_legacy_layout and not allow_raw_video:
            raise FileNotFoundError(
                f"Missing HMDB51 pre-extracted frame table: {frame_units_path}. "
                "Set allow_raw_video=True to use the legacy raw-video compatibility path explicitly."
            )
        if not frame_units_path.exists() and raw_path.exists():
            raise FileNotFoundError(f"Missing HMDB51 pre-extracted frame table: {frame_units_path}")
        if use_legacy_layout:
            lance_path = legacy_path
            extensions = [
                LanceTableExtension(
                    lance_path=str(root / "data" / "samples.lance"),
                    key="sample_id",
                    column_name="sample_id",
                    read_columns=["asset_id", "video_id", "video_path", "label_text", "negative_text", "class_name"],
                ),
                LanceTableExtension(
                    lance_path=str(root / "data" / "media.lance"),
                    key="asset_id",
                    column_name="asset_id",
                    read_columns=["video_bytes", "fps", "total_num_frames"],
                ),
            ]
        else:
            lance_path = raw_path
            extensions = [
                LanceTableExtension(
                    lance_path=str(frame_units_path),
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
            else build_hmdb51_train_default_transform(
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
        self.storage_layout = "legacy" if use_legacy_layout else "raw"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Hmdb51TrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("HMDB51 config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("HMDB51 config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "Hmdb51TrainDataset",
    "build_hmdb51_query_text",
    "build_hmdb51_train_default_transform",
    "transform_hmdb51_train_sample",
]
