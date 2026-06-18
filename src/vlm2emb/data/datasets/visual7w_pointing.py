"""Visual7W Pointing training dataset runtime."""

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

SUPPORTED_TASK_ARCHETYPES = ("image_visual_grounding",)
DEFAULT_QUERY_INSTRUCTION = "Select the portion of the image that answers the question"
DEFAULT_POSITIVE_INSTRUCTION = "Represent the given cropped image of the object."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
VISUAL7W_POINTING_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "visual_token_placement", "trailing_newline"},
    "positive": {"instruction", "visual_token_placement", "trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class Visual7wPointingTransformConfig:
    """Default transform configuration for Visual7W Pointing training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_visual_token_placement: str = "own_line"
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


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> Visual7wPointingTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="Visual7W-Pointing",
        allowed_keys=VISUAL7W_POINTING_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive)):
        if side_values.get("visual_token_placement") not in {None, "own_line"}:
            raise ValueError(f"Visual7W-Pointing transform.{side}.visual_token_placement must be 'own_line'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"Visual7W-Pointing transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("Visual7W-Pointing transform.negative.empty must be 'empty_multimodal_input'.")
    return Visual7wPointingTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="Visual7W-Pointing",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="Visual7W-Pointing transform.query",
            default="space",
        ),
        query_visual_token_placement="own_line",
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="Visual7W-Pointing",
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_visual_token_placement="own_line",
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _media_text(text: str, trailing_newline: str, *, dataset_name: str) -> str:
    """Attach a canonical image token to one media-bearing text side."""

    normalized = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN}\n{text}" if text else STANDARD_IMAGE_TOKEN,
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(normalized, trailing_newline, dataset_name=dataset_name)


def _build_query_text(record: Mapping[str, Any], config: Visual7wPointingTransformConfig) -> str:
    """Build the Visual7W Pointing query text."""

    body = normalize_text_whitespace(str(record.get("question", "") or ""))
    if body:
        body = f'"{body}"'
    text = join_instruction_body(
        config.query_instruction.strip(),
        body,
        separator=config.query_instruction_body_separator,
    )
    return _media_text(text, config.query_trailing_newline, dataset_name="Visual7W-Pointing")


def _build_media(record: Mapping[str, Any], *, image_field: str, path_field: str) -> list[dict[str, Any]]:
    """Decode one joined image field into runtime media."""

    image = record.get(image_field)
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError(f"Visual7W-Pointing row is missing required image bytes: {image_field}")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get(path_field, "") or "")},
        }
    ]


def transform_visual7w_pointing_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: Visual7wPointingTransformConfig,
) -> dict[str, Any]:
    """Transform one joined Visual7W Pointing row into a TrainSample."""

    return {
        "query": {
            "text": _build_query_text(record, config),
            "media": _build_media(record, image_field="query_image", path_field="query_image_path"),
        },
        "positive": {
            "text": _media_text(
                config.positive_instruction,
                config.positive_trailing_newline,
                dataset_name=dataset_name,
            ),
            "media": _build_media(record, image_field="positive_image", path_field="positive_image_path"),
        },
        "negative": {
            "text": _apply_trailing_newline("", config.negative_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "sample_id": str(record.get("sample_id", "") or ""),
            "qa_id": record.get("qa_id"),
            "image_id": record.get("image_id"),
            "image_filename": str(record.get("image_filename", "") or ""),
            "answer_box_id": record.get("answer_box_id"),
            "answer_name": str(record.get("answer_name", "") or ""),
            "question_type": str(record.get("question_type", "") or ""),
            "crop": {
                "x": record.get("crop_x"),
                "y": record.get("crop_y"),
                "width": record.get("crop_width"),
                "height": record.get("crop_height"),
            },
        },
    }


def build_visual7w_pointing_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default Visual7W Pointing transform."""

    return partial(
        transform_visual7w_pointing_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("visual7w_pointing_train")
class Visual7wPointingTrainDataset(SampleLanceDataset):
    """Visual7W Pointing image-to-crop grounding training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "Visual7W-Pointing",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("Visual7wPointingTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for Visual7wPointingTrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        root = Path(path)
        effective_transform = (
            transform
            if transform is not None
            else build_visual7w_pointing_train_default_transform(
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
                    key="query_image_path",
                    column_name="path",
                    read_columns=["image"],
                    column_name_map={"image": "query_image"},
                ),
                LanceTableExtension(
                    lance_path=str(root / "data" / "images.lance"),
                    key="positive_image_path",
                    column_name="path",
                    read_columns=["image"],
                    column_name_map={"image": "positive_image"},
                ),
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
    def from_config(cls, config: Mapping[str, Any]) -> Visual7wPointingTrainDataset:
        """Build a Visual7W Pointing dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "Visual7W-Pointing")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


LanceVisual7WPointingTrainDataset = Visual7wPointingTrainDataset

__all__ = [
    "Visual7wPointingTrainDataset",
    "LanceVisual7WPointingTrainDataset",
    "Visual7wPointingTransformConfig",
    "build_visual7w_pointing_train_default_transform",
    "transform_visual7w_pointing_train_sample",
]
