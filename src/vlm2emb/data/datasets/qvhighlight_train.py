"""QVHighlights training dataset runtime."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
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
from vlm2emb.data.datasets.video_frame_units import sample_prefixed_preextracted_frame_unit

SUPPORTED_TASK_ARCHETYPES = ("video_moment_retrieval",)
DEFAULT_QUERY_INSTRUCTION = "Find the clip that corresponds to the described scene in the given video:"
DEFAULT_POSITIVE_INSTRUCTION = "Understand the content of the provided video."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
WINDOW_SELECTION_VALUES = {"random", "first"}
QVHIGHLIGHT_TRANSFORM_KEYS = {
    "query": {
        "instruction",
        "instruction_body_separator",
        "visual_token_placement",
        "trailing_newline",
    },
    "positive": {"instruction", "visual_token_placement", "trailing_newline", "window_selection"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class QvhighlightTransformConfig:
    """Default transform configuration for QVHighlights training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_visual_token_placement: str = "own_line"
    query_trailing_newline: str = "ensure_single"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
    positive_visual_token_placement: str = "own_line"
    positive_trailing_newline: str = "ensure_single"
    positive_window_selection: str = "random"
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


def _normalize_visual_token_placement(value: Any, *, dataset_name: str, name: str) -> str:
    """Validate how a video token is placed for a media-bearing side."""

    if value is None:
        return "own_line"
    if value != "own_line":
        raise ValueError(f"{dataset_name} {name} must be 'own_line'.")
    return str(value)


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> QvhighlightTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="QVHighlights",
        allowed_keys=QVHIGHLIGHT_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("QVHighlights transform.negative.empty must be 'empty_multimodal_input'.")
    window_selection = positive.get("window_selection", "random")
    if window_selection not in WINDOW_SELECTION_VALUES:
        raise ValueError(
            f"QVHighlights transform.positive.window_selection must be one of {sorted(WINDOW_SELECTION_VALUES)}."
        )
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"QVHighlights transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    return QvhighlightTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="QVHighlights",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="QVHighlights",
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_visual_token_placement=_normalize_visual_token_placement(
            query.get("visual_token_placement"),
            dataset_name="QVHighlights",
            name="transform.query.visual_token_placement",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="QVHighlights",
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_visual_token_placement=_normalize_visual_token_placement(
            positive.get("visual_token_placement"),
            dataset_name="QVHighlights",
            name="transform.positive.visual_token_placement",
        ),
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        positive_window_selection=str(window_selection),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _media_text(text: str, *, trailing_newline: str) -> str:
    """Normalize a text side that carries video media."""

    normalized = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN} {text}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(normalized, trailing_newline, dataset_name="QVHighlights")


def build_qvhighlight_query_text(
    query: Any,
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the QVHighlights full-video query text."""

    body = normalize_text_whitespace(str(query or ""))
    text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _media_text(text, trailing_newline=trailing_newline)


def build_qvhighlight_positive_text(
    *,
    instruction: str = DEFAULT_POSITIVE_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the QVHighlights localized-clip positive text."""

    return _media_text(instruction, trailing_newline=trailing_newline)


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="QVHighlights"),
        "media": [],
    }


def _add_video_metadata(media_metadata: dict[str, Any], record: Mapping[str, Any]) -> None:
    """Attach source video metadata to one sampled media object."""

    fps = record.get("fps")
    total_num_frames = record.get("total_num_frames")
    if isinstance(fps, (int, float)) and fps > 0:
        media_metadata["fps"] = float(fps)
    if isinstance(total_num_frames, int) and total_num_frames >= 0:
        media_metadata.setdefault("source_total_num_frames", total_num_frames)
    if record.get("video_id") is not None:
        media_metadata["video_id"] = str(record["video_id"])
    if record.get("video") is not None:
        media_metadata["video"] = str(record["video"])


def _build_query_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample the query-side full-video frame-unit."""

    images, media_metadata = sample_prefixed_preextracted_frame_unit(
        record,
        prefix="query",
        num_frames=num_frames,
    )
    if record.get("vid") is not None:
        media_metadata["video_id"] = str(record["vid"])
    if not images:
        raise ValueError("QVHighlights row resolved to zero sampled query video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def _coerce_windows(value: Any) -> list[tuple[float, float]]:
    """Normalize Arrow/Python nested list windows."""

    if value is None:
        return []
    windows: list[tuple[float, float]] = []
    for window in value:
        if not isinstance(window, Sequence) or len(window) < 2:
            continue
        windows.append((float(window[0]), float(window[1])))
    return windows


def _select_window(record: Mapping[str, Any], *, mode: str) -> tuple[float, float]:
    """Select the localized positive window."""

    windows = _coerce_windows(record.get("relevant_windows"))
    if not windows:
        raise ValueError("QVHighlights training row has no relevant_windows for positive clip sampling.")
    if mode == "first":
        return windows[0]
    if mode == "random":
        return random.choice(windows)
    raise ValueError(f"Unsupported QVHighlights window_selection value: {mode}")


def _build_positive_media(
    record: Mapping[str, Any],
    *,
    num_frames: int,
    window_selection: str,
) -> list[dict[str, Any]]:
    """Read and sample the selected positive-side segment frame-unit."""

    images, media_metadata = sample_prefixed_preextracted_frame_unit(
        record,
        prefix="positive",
        num_frames=num_frames,
    )
    if record.get("vid") is not None:
        media_metadata["video_id"] = str(record["vid"])
    media_metadata["window_selection"] = window_selection
    if not images:
        raise ValueError("QVHighlights row resolved to zero sampled positive clip frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def _select_positive_frame_unit_key(record: Mapping[str, Any], *, mode: str) -> str:
    """Select one pre-extracted positive frame-unit key from a split row."""

    keys = record.get("positive_frame_unit_keys")
    if not isinstance(keys, list) or not keys:
        return ""
    if mode == "first":
        return str(keys[0])
    if mode == "random":
        return str(random.choice(keys))
    raise ValueError(f"Unsupported QVHighlights window_selection value: {mode}")


def transform_qvhighlight_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: QvhighlightTransformConfig,
) -> dict[str, Any]:
    """Transform one joined QVHighlights row into a TrainSample."""

    metadata = {
        key: value
        for key, value in {
            "dataset_name": dataset_name,
            "split": split,
            "qid": record.get("qid"),
            "vid": record.get("vid"),
            "video": record.get("video"),
            "duration": record.get("duration"),
        }.items()
        if value is not None
    }
    return {
        "query": {
            "text": build_qvhighlight_query_text(
                record.get("query"),
                instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_query_media(record, num_frames=num_frames),
        },
        "positive": {
            "text": build_qvhighlight_positive_text(
                instruction=config.positive_instruction,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": _build_positive_media(
                record,
                num_frames=num_frames,
                window_selection=config.positive_window_selection,
            ),
        },
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_qvhighlight_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default QVHighlights transform."""

    return partial(
        transform_qvhighlight_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("qvhighlight_train")
class QvhighlightTrainDataset(SampleLanceDataset):
    """QVHighlights video moment retrieval dataset over raw split rows."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        num_frames: int = 8,
        dataset_name: str = "QVHighlights",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("QvhighlightTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns.")
        root = Path(path)
        lance_path = root / "data" / f"{split}.lance"
        requested_columns = read_columns if read_columns is not None else columns
        normalized_config = _normalize_transform_kwargs(transform_kwargs)
        effective_transform = (
            transform
            if transform is not None
            else partial(
                transform_qvhighlight_train_sample,
                dataset_name=dataset_name,
                split=split,
                num_frames=num_frames,
                config=normalized_config,
            )
        )
        frame_unit_columns = [
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
        ]
        super().__init__(
            lance_path=str(lance_path),
            read_columns=requested_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "frames.lance"),
                    key="query_frame_unit_key",
                    column_name="frame_unit_key",
                    read_columns=frame_unit_columns,
                    column_name_map={column: f"query_{column}" for column in frame_unit_columns},
                ),
                LanceTableExtension(
                    lance_path=str(root / "data" / "frames.lance"),
                    key=partial(
                        _select_positive_frame_unit_key,
                        mode=normalized_config.positive_window_selection,
                    ),
                    column_name="frame_unit_key",
                    read_columns=frame_unit_columns,
                    column_name_map={column: f"positive_{column}" for column in frame_unit_columns},
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
    def from_config(cls, config: Mapping[str, Any]) -> QvhighlightTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("QVHighlights config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("QVHighlights config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_POSITIVE_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "QvhighlightTrainDataset",
    "build_qvhighlight_positive_text",
    "build_qvhighlight_query_text",
    "build_qvhighlight_train_default_transform",
    "transform_qvhighlight_train_sample",
]
