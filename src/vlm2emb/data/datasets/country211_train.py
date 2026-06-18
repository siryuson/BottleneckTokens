"""Country211 training dataset runtime."""

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
DEFAULT_QUERY_INSTRUCTION = "Identify the country depicted in the image"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
COUNTRY211_TRANSFORM_KEYS = {
    "query": {"instruction", "visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class Country211TransformConfig:
    """Default transform configuration for Country211 training rows."""

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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> Country211TransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="Country211",
        allowed_keys=COUNTRY211_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    if query.get("visual_token_placement") not in {None, "own_line"}:
        raise ValueError("Country211 transform.query.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("Country211 transform.negative.empty must be 'empty_multimodal_input'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"Country211 transform.{side}.trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
    return Country211TransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="Country211",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_visual_token_placement="own_line",
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_trailing_newline=positive.get("trailing_newline", "strip"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def build_country211_query_text(
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    trailing_newline: str = "ensure_single",
) -> str:
    """Build the Country211 image-to-country query text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN} {instruction}",
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="Country211")


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline, dataset_name="Country211"),
        "media": [],
    }


def _build_query_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the joined raw image bytes."""

    image = record.get("image")
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("Country211 row is missing required raw image bytes.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get("image_path", "") or "")},
        }
    ]


def transform_country211_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: Country211TransformConfig,
) -> dict[str, Any]:
    """Transform one joined Country211 row into a TrainSample."""

    class_name = str(record.get("class_name", "") or "")
    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "image_path": str(record.get("image_path", "") or ""),
        "source_tar": str(record.get("source_tar", "") or ""),
        "sample_key": str(record.get("sample_key", "") or ""),
        "class_index": record.get("class_index"),
        "class_name": class_name,
    }
    return {
        "query": {
            "text": build_country211_query_text(
                instruction=config.query_instruction,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": _build_query_media(record),
        },
        "positive": _text_side(class_name, config.positive_trailing_newline),
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": metadata,
    }


def build_country211_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default Country211 transform."""

    return partial(
        transform_country211_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("country211_train")
class Country211TrainDataset(SampleLanceDataset):
    """Country211 image-to-country training dataset over raw split rows."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "Country211",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("Country211TrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for Country211TrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        effective_transform = (
            transform
            if transform is not None
            else build_country211_train_default_transform(
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
    def from_config(cls, config: dict[str, Any]) -> Country211TrainDataset:
        """Build a Country211 dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "Country211")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "Country211TrainDataset",
    "Country211TransformConfig",
    "build_country211_query_text",
    "build_country211_train_default_transform",
    "transform_country211_train_sample",
]
