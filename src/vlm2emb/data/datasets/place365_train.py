"""Place365 training dataset runtime."""

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

SUPPORTED_TASK_ARCHETYPES = ("image_classification",)
DEFAULT_QUERY_INSTRUCTION = "Identify the scene shown in the image"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
PLACE365_TRANSFORM_KEYS = {
    "query": {"instruction", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class Place365TransformConfig:
    """Default transform configuration for Place365 training rows."""

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


def _normalize_visual_token_placement(value: Any) -> str:
    """Validate image token placement for Place365 query text."""

    if value is None:
        return "own_line"
    if value != "own_line":
        raise ValueError("Place365 transform.query.visual_token_placement must be 'own_line'.")
    return str(value)


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> Place365TransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="Place365",
        allowed_keys=PLACE365_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("Place365 transform.negative.empty must be 'empty_multimodal_input'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"Place365 transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    return Place365TransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="Place365",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_visual_token_placement=_normalize_visual_token_placement(query.get("visual_token_placement")),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def build_place365_query_text(
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the Place365 image-to-class query text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN} {instruction}",
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="Place365")


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="Place365"),
        "media": [],
    }


def _build_query_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the joined raw image bytes."""

    image = record.get("image")
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("Place365 row is missing required raw image bytes.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get("image_path", "") or "")},
        }
    ]


def transform_place365_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: Place365TransformConfig,
) -> dict[str, Any]:
    """Transform one joined Place365 row into a TrainSample."""

    class_name = str(record.get("class_name", "") or "")
    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "image_path": str(record.get("image_path", "") or ""),
        "class_name": class_name,
    }
    return {
        "query": {
            "text": build_place365_query_text(
                instruction=config.query_instruction,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_query_media(record),
        },
        "positive": _text_side(class_name, config.positive_trailing_newline),
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_place365_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default Place365 transform."""

    return partial(
        transform_place365_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("place365_train")
class Place365TrainDataset(SampleLanceDataset):
    """Place365 image-to-class training dataset over raw split rows."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "Place365",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("Place365TrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for Place365TrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        effective_transform = (
            transform
            if transform is not None
            else build_place365_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(Path(path) / "data" / f"{split}.lance"),
            read_columns=effective_read_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(Path(path) / "data" / "images.lance"),
                    key="image_path",
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
    def from_config(cls, config: dict[str, Any]) -> Place365TrainDataset:
        """Build a Place365 dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "Place365")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "Place365TrainDataset",
    "Place365TransformConfig",
    "build_place365_query_text",
    "build_place365_train_default_transform",
    "transform_place365_train_sample",
]
