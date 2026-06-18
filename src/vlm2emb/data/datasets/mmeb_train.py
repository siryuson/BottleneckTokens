"""MMEB-train parser registry for raw parquet training rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import LanceTableExtension, SampleLanceDataset, SampleTransform
from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    decode_image,
    normalize_text_whitespace,
)

MMEB_TRAIN_SUBSETS: tuple[str, ...] = (
    "A-OKVQA",
    "CIRR",
    "ChartQA",
    "DocVQA",
    "HatefulMemes",
    "ImageNet_1K",
    "InfographicsVQA",
    "MSCOCO",
    "MSCOCO_i2t",
    "MSCOCO_t2i",
    "N24News",
    "NIGHTS",
    "OK-VQA",
    "SUN397",
    "VOC2007",
    "VisDial",
    "Visual7W",
    "VisualNews_i2t",
    "VisualNews_t2i",
    "WebQA",
)
SUPPORTED_TASK_ARCHETYPES = (
    "classification",
    "question_answer",
    "retrieval",
    "grounding",
)


class MmebTrainSideContent(TypedDict):
    """Text and raw media key selected for one training side."""

    text: str
    path: str | None
    has_media: bool


class MmebTrainContent(TypedDict):
    """Parser-owned query/positive/negative content before media join."""

    subset: str
    query: MmebTrainSideContent
    positive: MmebTrainSideContent
    negative: MmebTrainSideContent


MmebTrainTransformInput = Mapping[str, Any] | None
MmebTrainTransformRules = dict[str, dict[str, str]]
MmebTrainParser = Callable[
    [Mapping[str, Any], str, MmebTrainTransformRules],
    MmebTrainContent,
]

_IMAGE_SIDE_TRANSFORM = {
    "visual_token_placement": "own_line",
    "trailing_newline": "ensure_single",
}
_EMPTY_NEGATIVE_TRANSFORM = {"empty": "empty_multimodal_input"}
_WEBQA_QUERY_TRANSFORM = {
    "visual_token_alignment": "strip_without_media",
    "trailing_newline": "ensure_single",
}
_VISDIAL_QUERY_TRANSFORM = {
    "instruction_body_separator": "newline",
    "trailing_newline": "normalize_blank_tail",
}
_VISDIAL_NEGATIVE_TRANSFORM = {
    "visual_token_alignment": "restore_if_empty",
    "visual_token_placement": "own_line",
    "trailing_newline": "ensure_single",
}
_CHARTQA_ANSWER_TRANSFORM = {
    "whitespace_normalization": "strip_trailing_horizontal",
}

DEFAULT_TRANSFORM_BY_SUBSET: dict[str, MmebTrainTransformRules] = {
    subset: {} for subset in MMEB_TRAIN_SUBSETS
}
for _subset in (
    "A-OKVQA",
    "ChartQA",
    "CIRR",
    "DocVQA",
    "HatefulMemes",
    "ImageNet_1K",
    "InfographicsVQA",
    "MSCOCO",
    "MSCOCO_i2t",
    "N24News",
    "NIGHTS",
    "OK-VQA",
    "SUN397",
    "VOC2007",
    "Visual7W",
    "VisualNews_i2t",
):
    DEFAULT_TRANSFORM_BY_SUBSET[_subset]["query"] = dict(_IMAGE_SIDE_TRANSFORM)
for _subset in (
    "CIRR",
    "MSCOCO",
    "MSCOCO_t2i",
    "NIGHTS",
    "VisualNews_t2i",
    "WebQA",
):
    DEFAULT_TRANSFORM_BY_SUBSET[_subset]["positive"] = dict(_IMAGE_SIDE_TRANSFORM)
for _subset in (
    "CIRR",
    "ImageNet_1K",
    "MSCOCO",
    "MSCOCO_i2t",
    "MSCOCO_t2i",
    "NIGHTS",
    "SUN397",
    "VOC2007",
    "VisualNews_i2t",
    "VisualNews_t2i",
    "WebQA",
):
    DEFAULT_TRANSFORM_BY_SUBSET[_subset]["negative"] = dict(_EMPTY_NEGATIVE_TRANSFORM)
DEFAULT_TRANSFORM_BY_SUBSET["WebQA"]["query"] = dict(_WEBQA_QUERY_TRANSFORM)
DEFAULT_TRANSFORM_BY_SUBSET["VisDial"] = {
    "query": dict(_VISDIAL_QUERY_TRANSFORM),
    "positive": dict(_IMAGE_SIDE_TRANSFORM),
    "negative": dict(_VISDIAL_NEGATIVE_TRANSFORM),
}
DEFAULT_TRANSFORM_BY_SUBSET["ChartQA"]["positive"] = dict(_CHARTQA_ANSWER_TRANSFORM)
DEFAULT_TRANSFORM_BY_SUBSET["ChartQA"]["negative"] = dict(_CHARTQA_ANSWER_TRANSFORM)

_ALLOWED_TRANSFORM_SIDES = {"query", "positive", "negative"}
_ALLOWED_SIDE_TRANSFORM_VALUES: dict[str, set[str]] = {
    "visual_token_alignment": {"strip_without_media", "restore_if_empty"},
    "visual_token_placement": {"own_line"},
    "trailing_newline": {
        "preserve",
        "strip",
        "ensure_single",
        "normalize_blank_tail",
    },
    "instruction_body_separator": {"newline"},
    "whitespace_normalization": {"strip_trailing_horizontal"},
    "empty": {"empty_multimodal_input"},
}


def _normalize_mmeb_train_transform(
    transform: MmebTrainTransformInput = None,
    *,
    subset: str,
) -> MmebTrainTransformRules:
    """Merge and validate YAML-friendly default transform kwargs."""

    rules = deepcopy(DEFAULT_TRANSFORM_BY_SUBSET[subset])
    if transform is None:
        return rules
    if not isinstance(transform, Mapping):
        raise TypeError("MMEB-train transform kwargs must be a mapping")

    unknown_sides = set(transform) - _ALLOWED_TRANSFORM_SIDES
    if unknown_sides:
        raise ValueError(f"Unsupported MMEB-train transform sides: {sorted(unknown_sides)}")

    for side, side_transform in transform.items():
        if side_transform is None:
            continue
        if not isinstance(side_transform, Mapping):
            raise TypeError(f"MMEB-train transform.{side} must be a mapping")
        side_rules = rules.setdefault(str(side), {})
        for key, value in side_transform.items():
            allowed_values = _ALLOWED_SIDE_TRANSFORM_VALUES.get(str(key))
            if allowed_values is None:
                raise ValueError(
                    f"Unsupported MMEB-train transform.{side} key: {key}"
                )
            string_value = str(value)
            if string_value not in allowed_values:
                raise ValueError(
                    f"Unsupported MMEB-train transform.{side}.{key} value: "
                    f"{string_value}"
                )
            side_rules[str(key)] = string_value
    return rules


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _path_value(value: Any) -> str | None:
    text = _string_value(value).strip()
    return text or None


def _query_image_path(record: Mapping[str, Any]) -> str | None:
    return _path_value(record.get("qry_image_path"))


def _positive_image_path(record: Mapping[str, Any]) -> str | None:
    return _path_value(record.get("pos_image_path"))


def _negative_image_path(record: Mapping[str, Any]) -> str | None:
    return _path_value(record.get("neg_image_path"))


def _replace_legacy_image_tokens(text: str) -> str:
    normalized = text
    for token in LEGACY_IMAGE_TOKENS:
        normalized = normalized.replace(token, STANDARD_IMAGE_TOKEN)
    return normalized


def _normalize_text_only_query_after_token_removal(text: str) -> str:
    """Normalize a text-only query after dropping an invalid visual token."""

    return normalize_text_whitespace(text.strip("\n"))


def _strip_horizontal_tail(text: str) -> str:
    """Remove accidental horizontal tail whitespace without touching newlines."""

    return text.rstrip(" \t")


def _ensure_single_trailing_newline(text: str) -> str:
    """Normalize one completed prompt block to exactly one trailing newline."""

    return normalize_text_whitespace(text, ensure_trailing_newline=True)


def _strip_trailing_newlines(text: str) -> str:
    """Remove only line-feed characters from the end of one prompt block."""

    return text.rstrip("\n")


def _normalize_visual_prompt_block(text: str) -> str:
    """Place a leading image token on its own line and close the prompt block."""

    if not text.startswith(STANDARD_IMAGE_TOKEN):
        return text
    body = text[len(STANDARD_IMAGE_TOKEN) :].lstrip(" \t\n")
    normalized = f"{STANDARD_IMAGE_TOKEN}\n{body}" if body else STANDARD_IMAGE_TOKEN
    return _ensure_single_trailing_newline(normalized)


def _apply_trailing_newline_transform(text: str, mode: str | None) -> str:
    """Apply the configured trailing-newline surface rule."""

    if mode is None or mode == "preserve":
        return text
    if mode == "strip":
        return _strip_trailing_newlines(text)
    if mode in {"ensure_single", "normalize_blank_tail"}:
        return _ensure_single_trailing_newline(text)
    raise ValueError(f"Unsupported MMEB-train trailing_newline value: {mode}")


def _strip_visual_tokens(text: str) -> str:
    """Remove standard and legacy visual tokens from one text string."""

    stripped = text.replace(STANDARD_IMAGE_TOKEN, "").replace("<image>", "")
    for token in LEGACY_IMAGE_TOKENS:
        stripped = stripped.replace(token, "")
    return _normalize_text_only_query_after_token_removal(stripped)


def _apply_side_transform(
    side: MmebTrainSideContent,
    side_transform: Mapping[str, str],
) -> MmebTrainSideContent:
    """Apply configured runtime-surface rules to one side."""

    text = side["text"]
    path = side["path"]
    has_media = side["has_media"]

    if side_transform.get("empty") == "empty_multimodal_input":
        return _side("", None)

    alignment = side_transform.get("visual_token_alignment")
    if alignment == "strip_without_media" and not has_media:
        text = _strip_visual_tokens(text)
    elif alignment == "restore_if_empty" and has_media and not text.strip():
        text = STANDARD_IMAGE_TOKEN

    if (
        side_transform.get("visual_token_placement") == "own_line"
        and has_media
    ):
        text = _normalize_visual_prompt_block(text)

    if side_transform.get("instruction_body_separator") == "newline":
        text = text.replace("image retrieval: Q:", "image retrieval:\nQ:", 1)

    if side_transform.get("whitespace_normalization") == "strip_trailing_horizontal":
        text = _strip_horizontal_tail(text)

    text = _apply_trailing_newline_transform(
        text,
        side_transform.get("trailing_newline"),
    )
    return _side(text, path)


def _side(text: str, path: str | None) -> MmebTrainSideContent:
    return {
        "text": text,
        "path": path,
        "has_media": path is not None,
    }


def _parse_origin_pair(
    raw_row: Mapping[str, Any],
    subset: str,
    transform_rules: MmebTrainTransformRules,
) -> MmebTrainContent:
    """Parse the origin MMEB-train raw pair fields into side content."""

    query_text = _replace_legacy_image_tokens(_string_value(raw_row.get("qry")))
    positive_text = _replace_legacy_image_tokens(_string_value(raw_row.get("pos_text")))
    negative_text = _replace_legacy_image_tokens(_string_value(raw_row.get("neg_text")))

    query_path = _path_value(raw_row.get("qry_image_path"))
    positive_path = _path_value(raw_row.get("pos_image_path"))
    negative_path = _path_value(raw_row.get("neg_image_path"))

    if (not query_text and not query_path) or (not positive_text and not positive_path):
        raise ValueError(f"MMEB-train row has empty query or positive side: subset={subset}")

    parsed: MmebTrainContent = {
        "subset": subset,
        "query": _side(query_text, query_path),
        "positive": _side(positive_text, positive_path),
        "negative": _side(negative_text, negative_path),
    }
    for side_name in ("query", "positive", "negative"):
        parsed[side_name] = _apply_side_transform(
            parsed[side_name],
            transform_rules.get(side_name, {}),
        )
    return parsed


def _parse_webqa(
    raw_row: Mapping[str, Any],
    subset: str,
    transform_rules: MmebTrainTransformRules,
) -> MmebTrainContent:
    return _parse_origin_pair(raw_row, subset, transform_rules)


def _parse_chartqa(
    raw_row: Mapping[str, Any],
    subset: str,
    transform_rules: MmebTrainTransformRules,
) -> MmebTrainContent:
    return _parse_origin_pair(raw_row, subset, transform_rules)


def _parse_visdial(
    raw_row: Mapping[str, Any],
    subset: str,
    transform_rules: MmebTrainTransformRules,
) -> MmebTrainContent:
    return _parse_origin_pair(raw_row, subset, transform_rules)


PARSER_BY_SUBSET: dict[str, MmebTrainParser] = {
    subset: _parse_origin_pair for subset in MMEB_TRAIN_SUBSETS
}
PARSER_BY_SUBSET["ChartQA"] = _parse_chartqa
PARSER_BY_SUBSET["WebQA"] = _parse_webqa
PARSER_BY_SUBSET["VisDial"] = _parse_visdial


def get_mmeb_train_parser(subset: str) -> MmebTrainParser:
    """Return the parser for one MMEB-train subset."""

    try:
        return PARSER_BY_SUBSET[subset]
    except KeyError as exc:
        raise KeyError(f"Unknown MMEB-train subset: {subset}") from exc


def parse_mmeb_train_content(
    raw_row: Mapping[str, Any],
    *,
    subset: str,
    transform: MmebTrainTransformInput = None,
) -> MmebTrainContent:
    """Parse one raw MMEB-train row using the subset registry."""

    parser = get_mmeb_train_parser(subset)
    transform_rules = _normalize_mmeb_train_transform(transform, subset=subset)
    return parser(raw_row, subset, transform_rules)


def _media_from_image_bytes(image: bytes | None, *, path: str | None) -> list[dict[str, Any]]:
    """Build one TrainSample media list from optional image bytes."""

    if path is None:
        return []
    if image is None:
        raise FileNotFoundError(f"MMEB-train image side-table row is missing for path={path}")
    return [
        {
            "kind": "image",
            "content": decode_image(image),
            "metadata": {"path": path},
        }
    ]


def _build_multimodal_side(
    side: MmebTrainSideContent,
    *,
    image: bytes | None,
) -> dict[str, Any]:
    """Build one TrainSample side from parsed content and joined image bytes."""

    path = side["path"]
    return {
        "text": side["text"],
        "media": _media_from_image_bytes(image, path=path),
    }


def transform_mmeb_train_sample(
    record: dict[str, Any],
    *,
    subset: str,
    split: str,
    dataset_name: str,
    transform_rules: MmebTrainTransformRules,
) -> dict[str, Any]:
    """Transform one joined MMEB-train raw record into a TrainSample."""

    parsed = parse_mmeb_train_content(
        record,
        subset=subset,
        transform=transform_rules,
    )
    return {
        "query": _build_multimodal_side(parsed["query"], image=record.get("query_image")),
        "positive": _build_multimodal_side(parsed["positive"], image=record.get("positive_image")),
        "negative": _build_multimodal_side(parsed["negative"], image=record.get("negative_image")),
        "metadata": {
            "dataset_name": dataset_name,
            "subset": subset,
            "split": split,
        },
    }


def build_mmeb_train_default_transform(
    *,
    subset: str,
    split: str,
    dataset_name: str,
    transform_kwargs: MmebTrainTransformInput = None,
) -> SampleTransform:
    """Build the default MMEB-train runtime transform."""

    transform_rules = _normalize_mmeb_train_transform(
        transform_kwargs,
        subset=subset,
    )
    return partial(
        transform_mmeb_train_sample,
        subset=subset,
        split=split,
        dataset_name=dataset_name,
        transform_rules=transform_rules,
    )


def build_mmeb_train_image_extensions(path: str | Path, *, subset: str) -> list[LanceTableExtension]:
    """Build query/positive/negative image side-table extensions."""

    image_table_path = str(Path(path) / "data" / "images" / f"{subset}.lance")
    return [
        LanceTableExtension(
            lance_path=image_table_path,
            key=_query_image_path,
            read_columns=["image"],
            column_name="path",
            missing="skip",
            column_name_map={"image": "query_image"},
        ),
        LanceTableExtension(
            lance_path=image_table_path,
            key=_positive_image_path,
            read_columns=["image"],
            column_name="path",
            missing="skip",
            column_name_map={"image": "positive_image"},
        ),
        LanceTableExtension(
            lance_path=image_table_path,
            key=_negative_image_path,
            read_columns=["image"],
            column_name="path",
            missing="skip",
            column_name_map={"image": "negative_image"},
        ),
    ]


@AutoDataset.register("mmeb_train")
class MmebTrainDataset(SampleLanceDataset):
    """MMEB-train runtime dataset over raw-preserving Lance tables."""

    def __init__(
        self,
        path: str,
        subset: str,
        split: str = "train",
        dataset_name: str = "mmeb_train",
        read_columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: MmebTrainTransformInput = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        self.path = path
        self.subset = subset
        self.split = split
        self.dataset_name = f"{dataset_name}/{subset}"
        self._images_path = str(Path(path) / "data" / "images" / f"{subset}.lance")
        lance_path = str(Path(path) / "data" / subset / f"{split}.lance")
        if transform is not None and not callable(transform):
            raise TypeError(
                "MmebTrainDataset transform must be a callable SampleTransform or None; "
                "use transform_kwargs for declarative default transform parameters"
            )

        effective_transform = (
            transform
            if transform is not None
            else build_mmeb_train_default_transform(
                subset=subset,
                split=split,
                dataset_name=self.dataset_name,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=lance_path,
            read_columns=read_columns,
            extensions=build_mmeb_train_image_extensions(path, subset=subset),
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MmebTrainDataset:
        """Build a MMEB-train dataset from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        config_transform = kwargs.pop("transform", None)
        if config_transform is not None:
            if callable(config_transform):
                kwargs["transform"] = config_transform
            else:
                kwargs["transform_kwargs"] = config_transform
        return cls(**kwargs)


__all__ = [
    "MMEB_TRAIN_SUBSETS",
    "PARSER_BY_SUBSET",
    "SUPPORTED_TASK_ARCHETYPES",
    "MmebTrainContent",
    "MmebTrainDataset",
    "MmebTrainParser",
    "MmebTrainSideContent",
    "MmebTrainTransformInput",
    "MmebTrainTransformRules",
    "build_mmeb_train_default_transform",
    "build_mmeb_train_image_extensions",
    "get_mmeb_train_parser",
    "parse_mmeb_train_content",
    "transform_mmeb_train_sample",
]
