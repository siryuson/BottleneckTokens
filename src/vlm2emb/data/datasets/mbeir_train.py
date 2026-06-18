"""M-BEIR training dataset runtime."""

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

SUPPORTED_TASK_ARCHETYPES = ("multimodal_retrieval", "image_retrieval")
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
INSTRUCTION_BODY_SEPARATOR_VALUES = {"space", "newline", "none"}
VISUAL_TOKEN_PLACEMENT_VALUES = {"own_line", "inline"}
EMPTY_NEGATIVE_VALUE = "empty_multimodal_input"

QUERY_INSTRUCTIONS: dict[str, str] = {
    "EDIS_train": "Find a news image that matches the provided caption:",
    "FashionIQ_train": "Find an image to match the fashion image and style note:",
    "OVEN_it2it_train": "Retrieve a Wikipedia image-description pair that provides evidence for the question of this image:",
    "OVEN_it2t_train": "Retrieve a Wikipedia paragraph that provides an answer to the given query about the image.",
    "INFOSEEK_it2it": "Retrieve a Wikipedia image-description pair that provides evidence for the question of this image:",
    "INFOSEEK_it2t": "Retrieve a Wikipedia paragraph that provides an answer to the given query about the image.",
    "Fashion200K_t2i": "Based on the following fashion description, retrieve the best matching image.",
    "Fashion200K_i2t": "Find a product description for the fashion item in the image.",
}
POSITIVE_INSTRUCTIONS: dict[str, str] = {
    "EDIS_train": "Represent the given image with related text information:",
    "FashionIQ_train": "Represent the given image.",
    "OVEN_it2it_train": "Represent the given Wikipedia image with related text information:",
    "OVEN_it2t_train": "",
    "INFOSEEK_it2it": "Represent the given Wikipedia image with related text information:",
    "INFOSEEK_it2t": "",
    "Fashion200K_t2i": "Represent the given image.",
    "Fashion200K_i2t": "",
}
MBEIR_TRANSFORM_KEYS = {
    "query": {
        "instruction",
        "instruction_body_separator",
        "visual_token_placement",
        "trailing_newline",
    },
    "positive": {
        "instruction",
        "instruction_body_separator",
        "visual_token_placement",
        "trailing_newline",
    },
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class MbeirTrainTransformConfig:
    """Default transform configuration for one M-BEIR training task."""

    query_instruction: str
    query_instruction_body_separator: str = "space"
    query_visual_token_placement: str = "own_line"
    query_trailing_newline: str = "ensure_single"
    positive_instruction: str = ""
    positive_instruction_body_separator: str = "space"
    positive_visual_token_placement: str = "own_line"
    positive_trailing_newline: str = "ensure_single"
    negative_trailing_newline: str = "ensure_single"


def first_positive_candidate_id(record: Mapping[str, Any]) -> str:
    """Resolve the positive candidate id used by the archive parser."""

    candidates = record.get("pos_cand_list") or []
    if not candidates:
        raise ValueError(f"M-BEIR row has no positive candidate: {record.get('qid')!r}")
    return str(candidates[0])


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
    """Join instruction and body text with a small explicit policy."""

    if separator not in INSTRUCTION_BODY_SEPARATOR_VALUES:
        raise ValueError(
            f"M-BEIR instruction_body_separator must be one of {sorted(INSTRUCTION_BODY_SEPARATOR_VALUES)}."
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


def _add_visual_token(text: str, placement: str, *, dataset_name: str) -> str:
    """Add the image token according to the dataset text policy."""

    if placement not in VISUAL_TOKEN_PLACEMENT_VALUES:
        raise ValueError(
            f"{dataset_name} visual_token_placement must be one of {sorted(VISUAL_TOKEN_PLACEMENT_VALUES)}."
        )
    separator = "\n" if placement == "own_line" else " "
    return canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN}{separator}{text}" if text else STANDARD_IMAGE_TOKEN,
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )


def _normalize_transform_kwargs(
    *,
    dataset_name: str,
    values: Mapping[str, Any] | None,
) -> MbeirTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    if dataset_name not in QUERY_INSTRUCTIONS:
        raise ValueError(f"Unsupported M-BEIR dataset_name: {dataset_name}")
    normalized = normalize_side_transform_mapping(
        values,
        dataset_name=dataset_name,
        allowed_keys=MBEIR_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})

    empty = negative.get("empty")
    if empty is not None and empty != EMPTY_NEGATIVE_VALUE:
        raise ValueError(f"{dataset_name} transform.negative.empty must be '{EMPTY_NEGATIVE_VALUE}'.")
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"{dataset_name} transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    for side, side_values in (("query", query), ("positive", positive)):
        separator = side_values.get("instruction_body_separator")
        if separator is not None and separator not in INSTRUCTION_BODY_SEPARATOR_VALUES:
            raise ValueError(
                f"{dataset_name} transform.{side}.instruction_body_separator must be one of "
                f"{sorted(INSTRUCTION_BODY_SEPARATOR_VALUES)}."
            )
        placement = side_values.get("visual_token_placement")
        if placement is not None and placement not in VISUAL_TOKEN_PLACEMENT_VALUES:
            raise ValueError(
                f"{dataset_name} transform.{side}.visual_token_placement must be one of "
                f"{sorted(VISUAL_TOKEN_PLACEMENT_VALUES)}."
            )

    return MbeirTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name=dataset_name,
            name="transform.query.instruction",
            default=QUERY_INSTRUCTIONS[dataset_name],
            allow_empty=False,
        ),
        query_instruction_body_separator=query.get("instruction_body_separator", "space"),
        query_visual_token_placement=query.get("visual_token_placement", "own_line"),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name=dataset_name,
            name="transform.positive.instruction",
            default=POSITIVE_INSTRUCTIONS[dataset_name],
            allow_empty=True,
        ),
        positive_instruction_body_separator=positive.get("instruction_body_separator", "space"),
        positive_visual_token_placement=positive.get("visual_token_placement", "own_line"),
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _media_from_image(record: Mapping[str, Any], *, image_field: str, path_field: str) -> list[dict[str, Any]]:
    """Build one image media list from an optional joined image field."""

    image = record.get(image_field)
    if image is None:
        return []
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError(f"M-BEIR joined field '{image_field}' must contain image bytes when present.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {"path": str(record.get(path_field, "") or "")},
        }
    ]


def _build_side_text(
    *,
    dataset_name: str,
    instruction: str,
    body: str,
    has_media: bool,
    instruction_body_separator: str,
    visual_token_placement: str,
    trailing_newline: str,
) -> str:
    """Build one query or positive text side."""

    text = _join_instruction_and_body(instruction, body, instruction_body_separator)
    text = normalize_text_whitespace(text)
    if has_media:
        text = _add_visual_token(text, visual_token_placement, dataset_name=dataset_name)
    return _apply_trailing_newline(text, trailing_newline, dataset_name=dataset_name)


def transform_mbeir_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: MbeirTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined M-BEIR row into a TrainSample."""

    query_media = _media_from_image(record, image_field="query_image", path_field="query_img_path")
    positive_media = _media_from_image(record, image_field="positive_image", path_field="positive_img_path")
    positive_text = str(record.get("positive_txt", "") or "")
    query_text = str(record.get("query_txt", "") or "")
    return {
        "query": {
            "text": _build_side_text(
                dataset_name=dataset_name,
                instruction=config.query_instruction,
                body=query_text,
                has_media=bool(query_media),
                instruction_body_separator=config.query_instruction_body_separator,
                visual_token_placement=config.query_visual_token_placement,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": query_media,
        },
        "positive": {
            "text": _build_side_text(
                dataset_name=dataset_name,
                instruction=config.positive_instruction,
                body=positive_text,
                has_media=bool(positive_media),
                instruction_body_separator=config.positive_instruction_body_separator,
                visual_token_placement=config.positive_visual_token_placement,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": positive_media,
        },
        "negative": {
            "text": _apply_trailing_newline("", config.negative_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "qid": str(record.get("qid", "") or ""),
            "task_id": record.get("task_id"),
            "query_modality": str(record.get("query_modality", "") or ""),
            "query_img_path": record.get("query_img_path"),
            "positive_did": first_positive_candidate_id(record),
            "positive_modality": str(record.get("positive_modality", "") or ""),
            "positive_img_path": record.get("positive_img_path"),
            "positive_src_content": record.get("positive_src_content"),
            "pos_cand_list": list(record.get("pos_cand_list") or []),
            "neg_cand_list": list(record.get("neg_cand_list") or []),
        },
    }


def build_mbeir_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default M-BEIR transform."""

    return partial(
        transform_mbeir_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(dataset_name=dataset_name, values=transform_kwargs),
    )


@AutoDataset.register("mbeir_train")
class MbeirTrainDataset(SampleLanceDataset):
    """M-BEIR training dataset over one archive-aligned task split."""

    def __init__(
        self,
        path: str,
        dataset_name: str,
        split: str = "official_train_without_mmeb_v2_eval",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("MbeirTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for MbeirTrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        root = Path(path)
        effective_transform = (
            transform
            if transform is not None
            else build_mbeir_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(root / "data" / dataset_name / f"{split}.lance"),
            read_columns=effective_read_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "candidates" / f"{dataset_name}.lance"),
                    key=first_positive_candidate_id,
                    column_name="did",
                    read_columns=["txt", "img_path", "modality", "src_content"],
                    column_name_map={
                        "txt": "positive_txt",
                        "img_path": "positive_img_path",
                        "modality": "positive_modality",
                        "src_content": "positive_src_content",
                    },
                ),
                LanceTableExtension(
                    lance_path=str(root / "data" / "images.lance"),
                    key="query_img_path",
                    column_name="path",
                    read_columns=["image"],
                    missing="skip",
                    column_name_map={"image": "query_image"},
                ),
                LanceTableExtension(
                    lance_path=str(root / "data" / "images.lance"),
                    key="positive_img_path",
                    column_name="path",
                    read_columns=["image"],
                    missing="skip",
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
    def from_config(cls, config: dict[str, Any]) -> MbeirTrainDataset:
        """Build a M-BEIR dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            dataset_name=str(config["dataset_name"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "MbeirTrainDataset",
    "MbeirTrainTransformConfig",
    "build_mbeir_train_default_transform",
    "first_positive_candidate_id",
    "transform_mbeir_train_sample",
]
