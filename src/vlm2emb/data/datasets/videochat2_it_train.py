"""VideoChat2-IT training dataset runtime."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import (
    LanceDataset,
    LanceKeyedRowResolver,
    SampleLanceDataset,
    SampleTransform,
)
from vlm2emb.data.datasets.const import (
    LEGACY_VIDEO_TOKENS,
    STANDARD_VIDEO_TOKEN,
    canonicalize_multimodal_text,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    normalize_side_transform_mapping,
)
from vlm2emb.data.datasets.video_frame_units import sample_preextracted_frame_unit

SUPPORTED_TASK_ARCHETYPES = ("split_video_instruction",)
DATASET_NAME = "VideoChat2-IT"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
VIDEOCHAT2_IT_TRANSFORM_KEYS = {
    "query": {"trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline"},
}
FRAME_READ_COLUMNS = [
    "frame_unit_key",
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
]


@dataclass(frozen=True)
class VideoChat2ItTransformConfig:
    """Default transform configuration for VideoChat2-IT rows."""

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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> VideoChat2ItTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name=DATASET_NAME,
        allowed_keys=VIDEOCHAT2_IT_TRANSFORM_KEYS,
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
    return VideoChat2ItTransformConfig(
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def build_videochat2_it_query_text(
    *,
    instruction: str,
    question: str,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the VideoChat2-IT video-conditioned query text."""

    body_parts = [
        normalize_text_whitespace(part)
        for part in (instruction, question)
        if normalize_text_whitespace(part)
    ]
    body = "\n".join(body_parts) or "Describe the provided video."
    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN}\n{body}",
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


def _build_query_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample a resolved pre-extracted frame unit."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    media_metadata["subset"] = str(record.get("subset", "") or "")
    media_metadata["frame_unit_source"] = str(record.get("frame_unit_source", "") or "")
    media_metadata["video"] = str(record.get("video", "") or "")
    if not images:
        raise ValueError("VideoChat2-IT row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_videochat2_it_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: VideoChat2ItTransformConfig,
) -> dict[str, Any]:
    """Transform one resolved VideoChat2-IT row into a TrainSample."""

    return {
        "query": {
            "text": build_videochat2_it_query_text(
                instruction=str(record.get("instruction", "") or ""),
                question=str(record.get("question", "") or ""),
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_query_media(record, num_frames=num_frames),
        },
        "positive": _text_side(str(record.get("answer", "") or ""), config.positive_trailing_newline),
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "id": record.get("id"),
            "subset": record.get("subset"),
            "source_family": record.get("source_family"),
            "task_view": record.get("task_view"),
            "video": record.get("video"),
            "frame_unit_source": record.get("frame_unit_source"),
            "frame_unit_key": record.get("frame_unit_key"),
            "source_relative_path": record.get("source_relative_path"),
            "source_row_index": record.get("source_row_index"),
            "qa_index": record.get("qa_index"),
        },
    }


def build_videochat2_it_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default VideoChat2-IT transform."""

    return partial(
        transform_videochat2_it_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("videochat2_it_train")
class VideoChat2ItTrainDataset(SampleLanceDataset):
    """VideoChat2-IT split video-instruction training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        num_frames: int = 8,
        dataset_name: str = DATASET_NAME,
        frame_roots: Mapping[str, str] | None = None,
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("VideoChat2ItTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for VideoChat2ItTrainDataset.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        resolved_frame_roots = dict(frame_roots or {})
        resolved_frame_roots.setdefault("activitynet_full", str(root / "data" / "activitynet_full_frames.lance"))
        effective_transform = (
            transform
            if transform is not None
            else build_videochat2_it_default_transform(
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(root / "data" / f"{split}.lance"),
            read_columns=requested_columns,
            extensions=[],
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.dataset_name = dataset_name
        self.split = split
        self.num_frames = num_frames
        self.frame_roots = resolved_frame_roots
        self._frame_datasets: dict[str, LanceDataset] = {}
        self._frame_resolvers: dict[str, LanceKeyedRowResolver] = {}

    def _get_frame_resolver(self, frame_source: str) -> LanceKeyedRowResolver:
        """Return the keyed frame-unit resolver for one source."""

        resolver = self._frame_resolvers.get(frame_source)
        if resolver is not None:
            return resolver
        path = self.frame_roots.get(frame_source)
        if not path:
            raise KeyError(f"Missing VideoChat2-IT frame root for source: {frame_source}")
        dataset = LanceDataset(path, lazy=self._lazy)
        self._frame_datasets[frame_source] = dataset
        resolver = LanceKeyedRowResolver(dataset, "frame_unit_key")
        self._frame_resolvers[frame_source] = resolver
        return resolver

    def _resolve_records(self, primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach frame-unit rows from the configured source-specific roots."""

        positions_by_source: dict[str, list[int]] = defaultdict(list)
        keys_by_source: dict[str, list[str]] = defaultdict(list)
        for index, row in enumerate(primary_rows):
            frame_source = str(row.get("frame_unit_source", "") or "")
            frame_key = str(row.get("frame_unit_key", "") or "")
            positions_by_source[frame_source].append(index)
            keys_by_source[frame_source].append(frame_key)

        resolved = [dict(row) for row in primary_rows]
        for frame_source, positions in positions_by_source.items():
            resolver = self._get_frame_resolver(frame_source)
            frame_rows = resolver.lookup_rows(
                keys_by_source[frame_source],
                columns=FRAME_READ_COLUMNS,
                missing="error",
            )
            for position, frame_row in zip(positions, frame_rows, strict=True):
                if frame_row is None:
                    raise KeyError(resolved[position].get("frame_unit_key"))
                resolved[position].update(frame_row)
        return resolved

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VideoChat2ItTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("VideoChat2-IT config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("VideoChat2-IT config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DATASET_NAME",
    "SUPPORTED_TASK_ARCHETYPES",
    "VideoChat2ItTrainDataset",
    "VideoChat2ItTransformConfig",
    "build_videochat2_it_default_transform",
    "build_videochat2_it_query_text",
    "transform_videochat2_it_train_sample",
]
