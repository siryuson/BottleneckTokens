"""TextVQA training dataset runtime."""

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
TEXTVQA_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class TextvqaTrainTransformConfig:
    """Default transform configuration for TextVQA training rows."""

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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> TextvqaTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(values, dataset_name="TextVQA", allowed_keys=TEXTVQA_TRANSFORM_KEYS)
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    if query.get("visual_token_placement") not in {None, "own_line"}:
        raise ValueError("TextVQA transform.query.visual_token_placement must be 'own_line'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"TextVQA transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("TextVQA transform.negative.empty must be 'empty_multimodal_input'.")
    return TextvqaTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="TextVQA",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=True,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="TextVQA transform.query",
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
    config: TextvqaTrainTransformConfig,
) -> str:
    """Build the TextVQA query text from the raw question."""

    text = join_instruction_body(
        config.query_instruction.strip(),
        normalize_text_whitespace(question),
        separator=config.query_instruction_body_separator,
    )
    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN}\n{text}" if text else STANDARD_IMAGE_TOKEN,
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, config.query_trailing_newline, dataset_name="TextVQA")


def _answer_text(record: Mapping[str, Any]) -> str:
    """Resolve the first TextVQA answer used by the archive parser."""

    answers = list(record.get("answers") or [])
    if not answers:
        raise ValueError(f"TextVQA row has no answers: {record.get('question_id')!r}")
    return str(answers[0])


def _build_query_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the joined TextVQA image bytes."""

    image = record.get("image")
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("TextVQA row is missing required image bytes.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get("image_id", "") or "")},
        }
    ]


def transform_textvqa_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: TextvqaTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined TextVQA row into a TrainSample."""

    positive_text = normalize_text_whitespace(_answer_text(record))
    return {
        "query": {
            "text": _build_query_text(question=str(record.get("question", "") or ""), config=config),
            "media": _build_query_media(record),
        },
        "positive": {
            "text": _apply_trailing_newline(positive_text, config.positive_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "negative": {
            "text": _apply_trailing_newline("", config.negative_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "image_id": str(record.get("image_id", "") or ""),
            "question_id": record.get("question_id"),
            "answers": list(record.get("answers") or []),
            "ocr_tokens": list(record.get("ocr_tokens") or []),
            "image_classes": list(record.get("image_classes") or []),
            "set_name": str(record.get("set_name", "") or ""),
        },
    }


def build_textvqa_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default TextVQA transform."""

    return partial(
        transform_textvqa_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("textvqa_train")
class TextvqaTrainDataset(SampleLanceDataset):
    """TextVQA image-question-to-answer training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "TextVQA",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("TextvqaTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for TextvqaTrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        root = Path(path)
        effective_transform = (
            transform
            if transform is not None
            else build_textvqa_train_default_transform(
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
                    key="image_id",
                    column_name="path",
                    read_columns=["image"],
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
    def from_config(cls, config: dict[str, Any]) -> TextvqaTrainDataset:
        """Build a TextVQA dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "TextVQA")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "TextvqaTrainDataset",
    "TextvqaTrainTransformConfig",
    "build_textvqa_train_default_transform",
    "transform_textvqa_train_sample",
]
