"""Wiki-SS-NQ training dataset runtime."""

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
    normalize_instruction_text,
    normalize_side_transform_mapping,
)

SUPPORTED_TASK_ARCHETYPES = ("image_retrieval",)
DEFAULT_QUERY_INSTRUCTION = "Find the document image that can answer the given query:"
DEFAULT_POSITIVE_INSTRUCTION = "Represent the given image"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
INSTRUCTION_BODY_SEPARATOR_VALUES = {"space", "newline", "none"}
WIKISSNQ_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"instruction", "visual_token_placement", "trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class WikissnqTrainTransformConfig:
    """Default transform configuration for Wiki-SS-NQ training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_trailing_newline: str = "ensure_single"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
    positive_visual_token_placement: str = "own_line"
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


def _join_instruction_and_body(instruction: str, body: str, separator: str) -> str:
    """Join instruction and body text with an explicit policy."""

    if separator not in INSTRUCTION_BODY_SEPARATOR_VALUES:
        raise ValueError(
            f"Wiki-SS-NQ instruction_body_separator must be one of {sorted(INSTRUCTION_BODY_SEPARATOR_VALUES)}."
        )
    instruction = instruction.strip()
    body = body.strip()
    if not instruction:
        return body
    if not body:
        return instruction
    if separator == "none":
        return f"{instruction}{body}"
    if separator == "newline":
        return f"{instruction}\n{body}"
    return f"{instruction} {body}"


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> WikissnqTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="Wiki-SS-NQ",
        allowed_keys=WIKISSNQ_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"Wiki-SS-NQ transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    separator = query.get("instruction_body_separator")
    if separator is not None and separator not in INSTRUCTION_BODY_SEPARATOR_VALUES:
        raise ValueError(
            "Wiki-SS-NQ transform.query.instruction_body_separator must be one of "
            f"{sorted(INSTRUCTION_BODY_SEPARATOR_VALUES)}."
        )
    placement = positive.get("visual_token_placement")
    if placement is not None and placement != "own_line":
        raise ValueError("Wiki-SS-NQ transform.positive.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("Wiki-SS-NQ transform.negative.empty must be 'empty_multimodal_input'.")
    return WikissnqTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="Wiki-SS-NQ",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=query.get("instruction_body_separator", "space"),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="Wiki-SS-NQ",
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_visual_token_placement="own_line",
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _positive_text(instruction: str, trailing_newline: str) -> str:
    """Build the positive document-image text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN}\n{instruction}",
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="Wiki-SS-NQ")


def _build_positive_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the joined screenshot bytes."""

    image = record.get("image")
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("Wiki-SS-NQ row is missing required screenshot bytes.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get("positive_docid", "") or "")},
        }
    ]


def transform_wikissnq_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: WikissnqTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined Wiki-SS-NQ row into a TrainSample."""

    query_text = normalize_text_whitespace(str(record.get("query", "") or ""))
    query_text = _join_instruction_and_body(
        config.query_instruction,
        query_text,
        config.query_instruction_body_separator,
    )
    query_text = _apply_trailing_newline(
        query_text,
        config.query_trailing_newline,
        dataset_name=dataset_name,
    )
    return {
        "query": {"text": query_text, "media": []},
        "positive": {
            "text": _positive_text(config.positive_instruction, config.positive_trailing_newline),
            "media": _build_positive_media(record),
        },
        "negative": {
            "text": _apply_trailing_newline("", config.negative_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "query_id": str(record.get("query_id", "") or ""),
            "positive_docid": str(record.get("positive_docid", "") or ""),
            "positive_rank": record.get("positive_rank"),
            "answers": list(record.get("answers") or []),
            "positive_title": str(record.get("positive_title", "") or ""),
        },
    }


def build_wikissnq_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default Wiki-SS-NQ transform."""

    return partial(
        transform_wikissnq_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("wikissnq_train")
class WikissnqTrainDataset(SampleLanceDataset):
    """Wiki-SS-NQ text-to-screenshot retrieval training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "Wiki-SS-NQ",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("WikissnqTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for WikissnqTrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        root = Path(path)
        effective_transform = (
            transform
            if transform is not None
            else build_wikissnq_train_default_transform(
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
                    key="positive_docid",
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
    def from_config(cls, config: dict[str, Any]) -> WikissnqTrainDataset:
        """Build a Wiki-SS-NQ dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "Wiki-SS-NQ")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "WikissnqTrainDataset",
    "WikissnqTrainTransformConfig",
    "build_wikissnq_train_default_transform",
    "transform_wikissnq_train_sample",
]
