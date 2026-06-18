from __future__ import annotations

"""MMEB image-to-text parser definitions.

These subsets are retrieval-style, so `origin` stays as close as possible to
the archive text formatting and `canonical` does not introduce extra rewrites
beyond approved image-token normalization.

This parser intentionally exposes no subset-specific style knobs: previous
query-* default keys were never consumed here and behaved like pseudo-config.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_image_media,
    extract_item_metadata,
    replace_legacy_image_tokens,
    strip_legacy_image_tokens,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem


DATASET_NAMES: tuple[str, ...] = (
    "MSCOCO_i2t",
    "VisualNews_i2t",
)

PATH_REL_BY_DATASET: dict[str, str] = {
    name: f"image-tasks/{name}"
    for name in DATASET_NAMES
}
MODALITY_BY_DATASET: dict[str, str] = {name: "image" for name in DATASET_NAMES}
TASK_TYPE_BY_DATASET: dict[str, str] = {
    name: "image_retrieval" for name in DATASET_NAMES
}

QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    name: ("id", "qry_text", "image") for name in DATASET_NAMES
}
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    name: ("id", "text") for name in DATASET_NAMES
}
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    name: dict(DEFAULT_TRANSFORM_KWARGS)
    for name in DATASET_NAMES
}

def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name not in DATASET_NAMES:
        raise KeyError(f"Unknown MMEB image_i2t dataset: {dataset_name}")


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the query-side raw columns for a dataset."""
    _require_dataset_name(dataset_name)
    return QUERY_READ_COLUMNS_BY_DATASET[dataset_name]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the candidate-side raw columns for a dataset."""
    _require_dataset_name(dataset_name)
    return CANDIDATE_READ_COLUMNS_BY_DATASET[dataset_name]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    """Return a copy of the dataset-local default transform kwargs."""
    _require_dataset_name(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])


def _string_scalar(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return _string_scalar(value[0])
    return "" if value is None else str(value)


def _transform_image_i2t_item(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    role: str,
    transform_kwargs: Mapping[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    if role not in {"query", "candidate"}:
        raise ValueError(f"Unsupported MMEB image_i2t role for {dataset_name}: {role}")
    del transform_kwargs

    has_media = bool(sample.get("images")) or sample.get("image") is not None
    raw_text = _string_scalar(sample.get("text"))
    qry_text = _string_scalar(sample.get("qry_text"))
    tgt_text = _string_scalar(sample.get("tgt_text"))

    if role == "candidate":
        text = raw_text or tgt_text
    elif has_media and qry_text:
        # Image-backed query rows already match the archive parser contract once
        # legacy markers are normalized to the standardized runtime token.
        text = replace_legacy_image_tokens(qry_text)
    elif qry_text:
        # Text-only fallbacks must drop visual placeholders so the runtime item
        # does not claim image input when no media slot exists.
        text = strip_legacy_image_tokens(qry_text)
    else:
        text = raw_text

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
    _require_dataset_name(dataset_name)
    normalized_transform_kwargs = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized_transform_kwargs.update(dict(transform_kwargs))
    return partial(
        _transform_image_i2t_item,
        dataset_name=dataset_name,
        role="query",
        transform_kwargs=normalized_transform_kwargs,
    )


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized_transform_kwargs = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized_transform_kwargs.update(dict(transform_kwargs))
    return partial(
        _transform_image_i2t_item,
        dataset_name=dataset_name,
        role="candidate",
        transform_kwargs=normalized_transform_kwargs,
    )


__all__ = [
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DATASET_NAMES",
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "DEFAULT_TRANSFORM_KWARGS",
    "MODALITY_BY_DATASET",
    "PATH_REL_BY_DATASET",
    "TASK_TYPE_BY_DATASET",
    "QUERY_READ_COLUMNS_BY_DATASET",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "get_query_read_columns",
    "build_candidate_transform",
    "build_query_transform",
]
