"""NExTQA training dataset runtime."""

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

SUPPORTED_TASK_ARCHETYPES = ("video_question_answer",)
DEFAULT_QUERY_INSTRUCTION = (
    "Given a video and a question, select the most accurate answer from the provided candidates. "
    "Return only the exact text of your chosen answer. Question:"
)
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
NEXTQA_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline"},
}


@dataclass(frozen=True)
class NextqaTrainTransformConfig:
    """Default transform configuration for NExTQA training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> NextqaTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="NExTQA",
        allowed_keys=NEXTQA_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"NExTQA transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    return NextqaTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="NExTQA",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="NExTQA",
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _options(record: Mapping[str, Any]) -> list[str]:
    """Return the five answer options from one converted row."""

    value = record.get("options")
    if isinstance(value, list) and len(value) == 5:
        return [normalize_text_whitespace(str(option)) for option in value]
    raise ValueError("NExTQA row must contain an options list with exactly five entries.")


def _answer_text(record: Mapping[str, Any]) -> str:
    """Resolve the positive answer option used by the archive parser."""

    answer = record.get("answer")
    if answer is None:
        raise ValueError("NExTQA row is missing answer.")
    options = _options(record)
    answer_index = int(answer)
    if answer_index < 0 or answer_index >= len(options):
        raise ValueError(f"NExTQA answer index {answer_index} is out of range for {len(options)} options.")
    return options[answer_index]


def _choice_block(question: str, options: list[str]) -> str:
    """Build the archive-style multiple-choice text block."""

    lines = [normalize_text_whitespace(question), "Options:"]
    for index, option in enumerate(options):
        lines.append(f"({chr(ord('A') + index)}) {option}")
    return "\n".join(lines).rstrip()


def build_nextqa_query_text(
    record: Mapping[str, Any],
    *,
    config: NextqaTrainTransformConfig,
) -> str:
    """Build the NExTQA video-question query text."""

    question = str(record.get("question", "") or "")
    body = _choice_block(question, _options(record))
    text = join_instruction_body(
        config.query_instruction,
        body,
        separator=config.query_instruction_body_separator,
    )
    text = canonicalize_multimodal_text(
        f"{STANDARD_VIDEO_TOKEN}\n{text}",
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, config.query_trailing_newline, dataset_name="NExTQA")


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="NExTQA"),
        "media": [],
    }


def _build_query_media(record: Mapping[str, Any], *, num_frames: int) -> list[dict[str, Any]]:
    """Read and sample the joined pre-extracted full-video frame-unit."""

    images, media_metadata = sample_preextracted_frame_unit(record, num_frames=num_frames)
    if record.get("video") is not None:
        media_metadata["video"] = str(record["video"])
    if record.get("video_path") is not None:
        media_metadata["video_path"] = str(record["video_path"])
    if not images:
        raise ValueError("NExTQA row resolved to zero sampled video frames.")
    return [{"kind": "video", "content": images, "metadata": media_metadata}]


def transform_nextqa_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    config: NextqaTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined NExTQA row into a TrainSample."""

    return {
        "query": {
            "text": build_nextqa_query_text(record, config=config),
            "media": _build_query_media(record, num_frames=num_frames),
        },
        "positive": _text_side(_answer_text(record), config.positive_trailing_newline),
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "video": record.get("video"),
            "video_path": record.get("video_path"),
            "qid": record.get("qid"),
            "type": record.get("type"),
            "answer": record.get("answer"),
            "options": list(_options(record)),
            "frame_count": record.get("frame_count"),
            "width": record.get("width"),
            "height": record.get("height"),
        },
    }


def build_nextqa_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    num_frames: int,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default NExTQA transform."""

    return partial(
        transform_nextqa_train_sample,
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("nextqa_train")
class NextqaTrainDataset(SampleLanceDataset):
    """NExTQA video-question-to-answer training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        num_frames: int = 8,
        dataset_name: str = "NExTQA",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("NextqaTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for NextqaTrainDataset.")
        root = Path(path)
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_nextqa_train_default_transform(
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
        self.dataset_name = dataset_name
        self.split = split
        self.num_frames = num_frames
        self.subset = None
        self.variant = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NextqaTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("NExTQA config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("NExTQA config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "NextqaTrainDataset",
    "NextqaTrainTransformConfig",
    "build_nextqa_query_text",
    "build_nextqa_train_default_transform",
    "transform_nextqa_train_sample",
]
