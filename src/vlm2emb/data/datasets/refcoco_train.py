"""RefCOCO training dataset runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from PIL import Image

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
DEFAULT_QUERY_INSTRUCTION = "Select the portion of the image that follows the language expressions:"
DEFAULT_POSITIVE_INSTRUCTION = "Represent the given cropped image of the object."
DATASET_NAME = "RefCOCO"
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
CAPTION_SELECTION_VALUES = {"first", "all_joined"}
REFCOCO_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline", "caption_selection"},
    "positive": {"instruction", "instruction_body_separator", "trailing_newline", "caption_selection"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class RefcocoTrainTransformConfig:
    """Default transform configuration for RefCOCO training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_trailing_newline: str = "ensure_single"
    query_caption_selection: str = "first"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
    positive_instruction_body_separator: str = "space"
    positive_trailing_newline: str = "ensure_single"
    positive_caption_selection: str | None = None
    negative_trailing_newline: str = "ensure_single"


def _apply_trailing_newline(text: str, mode: str) -> str:
    """Apply one side's trailing-newline policy."""

    if mode not in TRAILING_NEWLINE_VALUES:
        raise ValueError(f"{DATASET_NAME} trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}.")
    if mode == "preserve":
        return text
    if mode == "strip":
        return text.rstrip("\n")
    return text.rstrip("\n") + "\n" if text else ""


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> RefcocoTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name=DATASET_NAME,
        allowed_keys=REFCOCO_TRANSFORM_KEYS,
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
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError(f"{DATASET_NAME} transform.negative.empty must be 'empty_multimodal_input'.")
    query_caption_selection = query.get("caption_selection", "first")
    if query_caption_selection not in CAPTION_SELECTION_VALUES:
        raise ValueError(
            f"{DATASET_NAME} transform.query.caption_selection must be one of {sorted(CAPTION_SELECTION_VALUES)}."
        )
    positive_caption_selection = positive.get("caption_selection")
    if positive_caption_selection is not None and positive_caption_selection not in CAPTION_SELECTION_VALUES:
        raise ValueError(
            f"{DATASET_NAME} transform.positive.caption_selection must be one of "
            f"{sorted(CAPTION_SELECTION_VALUES)}."
        )
    return RefcocoTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name=DATASET_NAME,
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name=DATASET_NAME,
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        query_caption_selection=query_caption_selection,
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name=DATASET_NAME,
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_instruction_body_separator=normalize_instruction_body_separator(
            positive.get("instruction_body_separator"),
            dataset_name=DATASET_NAME,
            name="transform.positive.instruction_body_separator",
            default="space",
        ),
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        positive_caption_selection=positive_caption_selection,
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _extension_key(record: Mapping[str, Any]) -> int:
    """Resolve the COCO image table key."""

    return int(record["image_id"])


def _select_caption(value: Any, *, selection: str) -> str:
    """Return one referring expression from source rows."""

    captions = [str(caption) for caption in value or [] if caption is not None and str(caption).strip()]
    if not captions:
        return ""
    if selection == "first":
        return captions[0]
    if selection == "all_joined":
        return "; ".join(captions)
    raise ValueError(f"{DATASET_NAME} caption_selection must be one of {sorted(CAPTION_SELECTION_VALUES)}.")


def _image_text(text: str, trailing_newline: str) -> str:
    """Attach a canonical image token to one media-bearing text side."""

    normalized = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN}\n{text}" if text else STANDARD_IMAGE_TOKEN,
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(normalized, trailing_newline)


def build_refcoco_query_text(
    captions: Any,
    *,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
    caption_selection: str = "first",
) -> str:
    """Build the RefCOCO image grounding query text."""

    caption = normalize_text_whitespace(_select_caption(captions, selection=caption_selection))
    body = f'"{caption}"' if caption else ""
    text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _image_text(text, trailing_newline)


def build_refcoco_positive_text(
    captions: Any = None,
    *,
    instruction: str = DEFAULT_POSITIVE_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str = "ensure_single",
    caption_selection: str | None = None,
) -> str:
    """Build the RefCOCO positive crop text."""

    if caption_selection is None:
        text = instruction
    else:
        caption = normalize_text_whitespace(_select_caption(captions, selection=caption_selection))
        body = f'"{caption}"' if caption else ""
        text = join_instruction_body(instruction, body, separator=instruction_body_separator)
    return _image_text(text, trailing_newline)


def _text_side(text: str, trailing_newline: str) -> dict[str, Any]:
    """Build one text-only TrainSample side."""

    normalized = normalize_text_whitespace(text)
    return {
        "text": _apply_trailing_newline(normalized, trailing_newline),
        "media": [],
    }


def _decode_full_image(record: Mapping[str, Any]) -> Image.Image:
    """Decode the joined COCO image bytes."""

    image = record.get("image")
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("RefCOCO row is missing required COCO image bytes.")
    return decode_image(bytes(image))


def _crop_positive_image(image: Image.Image, bbox: Any) -> Image.Image:
    """Crop a positive object box from the full image."""

    values = [float(value) for value in bbox or []]
    if len(values) != 4:
        raise ValueError("RefCOCO bbox must contain [x1, y1, x2, y2].")
    width, height = image.size
    left = max(0.0, min(values[0], float(width)))
    top = max(0.0, min(values[1], float(height)))
    right = max(left + 1.0, min(values[2], float(width)))
    bottom = max(top + 1.0, min(values[3], float(height)))
    return image.crop((left, top, right, bottom))


def transform_refcoco_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: RefcocoTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined RefCOCO row into a TrainSample."""

    full_image = _decode_full_image(record)
    crop = _crop_positive_image(full_image, record.get("bbox"))
    return {
        "query": {
            "text": build_refcoco_query_text(
                record.get("captions"),
                instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
                caption_selection=config.query_caption_selection,
            ),
            "media": [{"kind": "image", "content": full_image, "metadata": {"image_id": record.get("image_id")}}],
        },
        "positive": {
            "text": build_refcoco_positive_text(
                record.get("captions"),
                instruction=config.positive_instruction,
                instruction_body_separator=config.positive_instruction_body_separator,
                trailing_newline=config.positive_trailing_newline,
                caption_selection=config.positive_caption_selection,
            ),
            "media": [
                {
                    "kind": "image",
                    "content": crop,
                    "metadata": {
                        "ann_id": record.get("ann_id"),
                        "image_id": record.get("image_id"),
                        "bbox": record.get("bbox"),
                    },
                }
            ],
        },
        "negative": _text_side("", config.negative_trailing_newline),
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "ann_id": record.get("ann_id"),
            "ref_id": record.get("ref_id"),
            "image_id": record.get("image_id"),
            "source_split": record.get("split"),
            "captions": record.get("captions"),
            "bbox": record.get("bbox"),
            "category_id": record.get("category_id"),
            "global_image_id": record.get("global_image_id"),
            "anns_id": record.get("anns_id"),
        },
    }


def build_refcoco_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default RefCOCO transform."""

    return partial(
        transform_refcoco_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("refcoco_train")
class RefcocoTrainDataset(SampleLanceDataset):
    """RefCOCO image-to-crop visual grounding training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = DATASET_NAME,
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("RefcocoTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for RefcocoTrainDataset.")
        requested_columns = read_columns if read_columns is not None else columns
        effective_transform = (
            transform
            if transform is not None
            else build_refcoco_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=f"{path}/data/{split}.lance",
            read_columns=requested_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=f"{path}/data/images.lance",
                    key=_extension_key,
                    column_name="image_id",
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
    def from_config(cls, config: Mapping[str, Any]) -> RefcocoTrainDataset:
        """Build from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        kwargs.pop("weight", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("RefCOCO config transform must be mapping, callable, or None.")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("RefCOCO config cannot set both transform and transform_kwargs.")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DEFAULT_POSITIVE_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "SUPPORTED_TASK_ARCHETYPES",
    "RefcocoTrainDataset",
    "RefcocoTrainTransformConfig",
    "build_refcoco_positive_text",
    "build_refcoco_query_text",
    "build_refcoco_train_default_transform",
    "transform_refcoco_train_sample",
]
