"""ScienceQA training dataset runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import LanceTableExtension, SampleLanceDataset, SampleTransform
from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    canonicalize_multimodal_text,
    decode_image,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    join_instruction_body,
    normalize_instruction_body_separator,
    normalize_instruction_text,
    normalize_side_transform_mapping,
)

SUPPORTED_TASK_ARCHETYPES = ("image_question_answer",)
DEFAULT_QUERY_INSTRUCTION = "Represent the given image with the following question:"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
SCIENCEQA_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class ScienceqaTrainTransformConfig:
    """Default transform configuration for ScienceQA training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> ScienceqaTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="ScienceQA",
        allowed_keys=SCIENCEQA_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    if query.get("visual_token_placement") not in {None, "own_line"}:
        raise ValueError("ScienceQA transform.query.visual_token_placement must be 'own_line'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"ScienceQA transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("ScienceQA transform.negative.empty must be 'empty_multimodal_input'.")
    return ScienceqaTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="ScienceQA",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=True,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="ScienceQA transform.query",
            default="space",
        ),
        query_visual_token_placement="own_line",
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _build_query_text(
    *,
    question: str,
    has_media: bool,
    config: ScienceqaTrainTransformConfig,
) -> str:
    """Build the ScienceQA query text from the raw question."""

    text = join_instruction_body(
        config.query_instruction.strip(),
        normalize_text_whitespace(question),
        separator=config.query_instruction_body_separator,
    )
    if has_media:
        text = canonicalize_multimodal_text(
            f"{STANDARD_IMAGE_TOKEN}\n{text}" if text else STANDARD_IMAGE_TOKEN,
            token=STANDARD_IMAGE_TOKEN,
            legacy_tokens=LEGACY_IMAGE_TOKENS,
        )
    return _apply_trailing_newline(text, config.query_trailing_newline, dataset_name="ScienceQA")


def _answer_text(record: Mapping[str, Any]) -> str:
    """Resolve the positive answer choice used by the archive parser."""

    choices = list(record.get("choices") or [])
    answer = record.get("answer")
    if answer is None:
        raise ValueError("ScienceQA row is missing answer.")
    answer_index = int(answer)
    if answer_index < 0 or answer_index >= len(choices):
        raise ValueError(f"ScienceQA answer index {answer_index} is out of range for {len(choices)} choices.")
    return str(choices[answer_index])


def _build_query_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the optional joined ScienceQA image."""

    image = record.get("image")
    if image is None:
        return []
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("ScienceQA joined image field must contain image bytes when present.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get("image_path", "") or "")},
        }
    ]


def transform_scienceqa_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: ScienceqaTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined ScienceQA row into a TrainSample."""

    query_media = _build_query_media(record)
    positive_text = normalize_text_whitespace(_answer_text(record))
    return {
        "query": {
            "text": _build_query_text(
                question=str(record.get("question", "") or ""),
                has_media=bool(query_media),
                config=config,
            ),
            "media": query_media,
        },
        "positive": {
            "text": _apply_trailing_newline(
                positive_text,
                config.positive_trailing_newline,
                dataset_name=dataset_name,
            ),
            "media": [],
        },
        "negative": {
            "text": _apply_trailing_newline("", config.negative_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "image_path": record.get("image_path"),
            "answer": int(record.get("answer")),
            "choices": list(record.get("choices") or []),
            "task": record.get("task"),
            "grade": record.get("grade"),
            "subject": record.get("subject"),
            "topic": record.get("topic"),
            "category": record.get("category"),
            "skill": record.get("skill"),
        },
    }


def build_scienceqa_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default ScienceQA transform."""

    return partial(
        transform_scienceqa_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("scienceqa_train")
class ScienceqaTrainDataset(SampleLanceDataset):
    """ScienceQA image-question-to-answer training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "ScienceQA",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("ScienceqaTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for ScienceqaTrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        root = Path(path)
        effective_transform = (
            transform
            if transform is not None
            else build_scienceqa_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(root / "data" / f"{split}.lance"),
            read_columns=effective_read_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "images.lance"),
                    key="image_path",
                    column_name="path",
                    read_columns=["image"],
                    missing="skip",
                )
            ],
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.dataset_name = dataset_name
        self.split = split
        self.subset = None
        self.variant = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ScienceqaTrainDataset:
        """Build a ScienceQA dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "ScienceQA")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "ScienceqaTrainDataset",
    "ScienceqaTrainTransformConfig",
    "build_scienceqa_train_default_transform",
    "transform_scienceqa_train_sample",
]
