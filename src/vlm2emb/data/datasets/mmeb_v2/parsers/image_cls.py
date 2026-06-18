from __future__ import annotations

"""MMEB image classification parser definitions.

This module keeps the image-classification family self-contained:
- each dataset declares its own read columns and default transform kwargs;
- `origin` keeps the archive prompt/layout behavior;
- `canonical` only normalizes the approved image-token and newline handling
  controlled by the per-dataset default kwargs and runtime overrides.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_image_media,
    extract_item_metadata,
    compose_parser_text_with_instruction,
    replace_legacy_image_tokens,
    strip_legacy_image_tokens,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem


def _image_cls_default_transform_kwargs() -> dict[str, Any]:
    """Return dataset-local prompt defaults for archive-aligned image queries.

    These kwargs stay parser-local on purpose. They describe the approved
    image-classification surface without introducing a shared rule engine:
    - `query_replace_space_newline=True` keeps the visual token on its own line;
    - `query_append_trailing_newline=True` keeps the parser-owned terminal
      newline for the complete prompt block in both `origin` and `canonical`.
    """
    return {
        "runtime_mode": "canonical",
        "query_instruction_leading_newline": False,
        "query_replace_space_newline": True,
        "query_append_trailing_newline": True,
        "query_append_extra_blank_line": False,
    }


DATASET_NAMES: tuple[str, ...] = (
    "HatefulMemes",
    "VOC2007",
    "SUN397",
    "Place365",
    "ImageNet-A",
    "ImageNet-R",
    "ObjectNet",
    "Country211",
    "N24News",
    "ImageNet-1K",
)

QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    name: ("id", "qry_text", "image") for name in DATASET_NAMES
}
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    name: ("id", "text") for name in DATASET_NAMES
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    name: _image_cls_default_transform_kwargs() for name in DATASET_NAMES
}

PATH_REL_BY_DATASET: dict[str, str] = {name: f"image-tasks/{name}" for name in DATASET_NAMES}


def _require_dataset_name(dataset_name: str) -> str:
    if dataset_name not in QUERY_READ_COLUMNS_BY_DATASET:
        raise KeyError(f"Unknown MMEB image_cls dataset: {dataset_name}")
    return dataset_name


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the query-side raw columns for a dataset."""
    return QUERY_READ_COLUMNS_BY_DATASET[_require_dataset_name(dataset_name)]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the candidate-side raw columns for a dataset."""
    return CANDIDATE_READ_COLUMNS_BY_DATASET[_require_dataset_name(dataset_name)]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    """Return a copy of the dataset-local default transform kwargs."""
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[_require_dataset_name(dataset_name)])


def _image_bool_default(defaults: Mapping[str, Any], key: str, fallback: bool) -> bool:
    value = defaults.get(key)
    return fallback if value is None else bool(value)


def _string_scalar(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return _string_scalar(value[0])
    return "" if value is None else str(value)


def _clean_instruction(raw_instruction: str, *, leading_newline: bool = False) -> str:
    # Instructions never keep visual placeholders as text content. When an
    # upstream artifact still stores a legacy token in the instruction field,
    # drop it here so the parser remains the single owner of token placement.
    cleaned = replace_legacy_image_tokens(raw_instruction, replacement="").strip()
    if cleaned and leading_newline:
        return "\n" + cleaned
    return cleaned


def _compose_query_with_instruction(
    *,
    qry_inst: str,
    qry_text: str,
    defaults: Mapping[str, Any],
) -> str:
    # Image-classification prompts keep the parser-owned trailing newline in
    # both `origin` and `canonical`. The earlier parser-strip/loader-reappend
    # split was only a temporary compatibility bridge. Keeping the newline here
    # makes the parser the single owner of the final surface form again.
    return compose_parser_text_with_instruction(
        qry_text,
        instruction=_clean_instruction(
            qry_inst,
            leading_newline=_image_bool_default(defaults, "query_instruction_leading_newline", False),
        ),
        add_image_token=True,
        replace_space_newline=_image_bool_default(defaults, "query_replace_space_newline", True),
        append_trailing_newline=_image_bool_default(defaults, "query_append_trailing_newline", True),
        append_extra_blank_line=_image_bool_default(defaults, "query_append_extra_blank_line", False),
    )


def _compose_image_cls_text(
    sample: dict[str, Any],
    *,
    role: str,
    transform_kwargs: Mapping[str, Any],
) -> str:
    has_media = bool(sample.get("images")) or sample.get("image") is not None
    raw_text = _string_scalar(sample.get("text"))
    qry_text = _string_scalar(sample.get("qry_text"))
    qry_inst = _string_scalar(sample.get("qry_inst"))

    if role == "candidate":
        return raw_text

    if has_media and qry_inst:
        # Image-backed classification queries own their final layout here:
        # the parser inserts the standardized image token and preserves the
        # archive-aligned terminal newline instead of relying on loader fixes.
        return _compose_query_with_instruction(
            qry_inst=qry_inst,
            qry_text=qry_text,
            defaults=transform_kwargs,
        )
    if has_media and qry_text:
        # The rerun Lance artifacts still carry legacy image tokens, but the
        # original parser surface already matched the standardized Qwen token.
        # Keep this normalization enabled for both `origin` and `canonical` so
        # runtime text follows the parser contract rather than the raw storage
        # artifact. This can be removed only after conversion outputs are
        # rewritten to the standardized token surface and old eval baselines are
        # regenerated against those new artifacts.
        return replace_legacy_image_tokens(qry_text)
    if qry_text:
        # Text-only fallbacks cannot claim visual input. This is the same
        # visual-token/media alignment invariant used across the runtime stack.
        return strip_legacy_image_tokens(qry_text)
    return raw_text


def _transform_image_cls_item(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    role: str,
    transform_kwargs: Mapping[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    if role not in {"query", "candidate"}:
        raise ValueError(f"Unsupported MMEB image_cls role for {dataset_name}: {role}")

    text = _compose_image_cls_text(sample, role=role, transform_kwargs=transform_kwargs)
    multimodal_input = MultiModalInput(text=text, media=collect_image_media(sample))
    consumed_fields = {
        "id",
        "text",
        "images",
        "image",
        "qry_text",
        "qry_inst",
        "qry_img_path",
        "tgt_text",
        "tgt_inst",
        "tgt_img_path",
        "media_metadata",
    }
    metadata = extract_item_metadata(sample, consumed_fields=consumed_fields)
    if role == "query":
        item: RetrievalQueryItem = {"id": str(sample["id"]), "query": multimodal_input}
    else:
        item = {"id": str(sample["id"]), "candidate": multimodal_input}
    if metadata:
        item["metadata"] = metadata
    return item


def build_query_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    dataset_name = _require_dataset_name(dataset_name)
    normalized_transform_kwargs = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized_transform_kwargs.update(dict(transform_kwargs))
    return partial(
        _transform_image_cls_item,
        dataset_name=dataset_name,
        role="query",
        transform_kwargs=normalized_transform_kwargs,
    )


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    dataset_name = _require_dataset_name(dataset_name)
    normalized_transform_kwargs = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized_transform_kwargs.update(dict(transform_kwargs))
    return partial(
        _transform_image_cls_item,
        dataset_name=dataset_name,
        role="candidate",
        transform_kwargs=normalized_transform_kwargs,
    )


__all__ = [
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DATASET_NAMES",
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "PATH_REL_BY_DATASET",
    "QUERY_READ_COLUMNS_BY_DATASET",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "get_query_read_columns",
    "build_candidate_transform",
    "build_query_transform",
]
