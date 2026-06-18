from __future__ import annotations

"""MMEB parser for SmthSmthV2 video classification.

The parser keeps the archive-style dataset instruction and read-column contract
local so origin matches the original prompt shape. Canonical keeps the wording
but makes the video token a standalone line and ensures the full prompt ends
with one trailing newline.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_video_media,
    extract_item_metadata,
    compose_parser_text_with_instruction,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

DATASET_NAME = "SmthSmthV2"
PATH_REL = "video-tasks/SmthSmthV2"
MODALITY = "video"
TASK_TYPE = "video_classification"
# Dataset-local read columns keep the classification query/candidate contract
# explicit for the parser review.
QUERY_READ_COLUMNS = ("id", "pos_text", "video", "video_id")
CANDIDATE_READ_COLUMNS = ("id", "text")
# Keep the prompt defaults alongside the dataset because the wording is part of
# the archive-compatible runtime contract.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "classification_instruction": "What actions or object interactions are being performed by the person in the video?",
    "classification_token_separator_origin": " ",
    "classification_token_separator_canonical": "\n",
    # Canonical keeps the archive wording but owns both the newline-separated
    # visual token layout and the trailing newline directly in the parser so
    # it no longer depends on the loader patch.
    "classification_append_trailing_newline_origin": False,
    "classification_append_trailing_newline_canonical": True,
}

def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported parser dataset: {dataset_name}")


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return QUERY_READ_COLUMNS


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return CANDIDATE_READ_COLUMNS


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    _require_dataset_name(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS)


def _normalize_transform_kwargs(transform_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # Dataset-local defaults are the baseline; callers may override per run.
    normalized = dict(DEFAULT_TRANSFORM_KWARGS)
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def _runtime_mode(transform_kwargs: Mapping[str, Any]) -> str:
    return str(transform_kwargs.get("runtime_mode") or "canonical")


def _classification_append_trailing_newline(transform_kwargs: Mapping[str, Any]) -> bool:
    explicit = transform_kwargs.get("classification_append_trailing_newline")
    if explicit is not None:
        return bool(explicit)
    return bool(
        transform_kwargs.get(
            f"classification_append_trailing_newline_{_runtime_mode(transform_kwargs)}",
            False,
        )
    )


def _classification_token_separator(transform_kwargs: Mapping[str, Any]) -> str:
    explicit = transform_kwargs.get("classification_token_separator")
    if explicit is not None:
        return str(explicit)
    return str(
        transform_kwargs.get(
            f"classification_token_separator_{_runtime_mode(transform_kwargs)}",
            " ",
        )
    )


def _transform_ssv2_item(
    sample: dict[str, Any],
    *,
    role: str,
    transform_kwargs: dict[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    if role == "query":
        # Origin and canonical share the archive wording. Canonical only owns a
        # parser-local layout upgrade: the video token moves to its own line and
        # the complete prompt ends with one newline. Keeping this in the parser
        # avoids another loader-wide text rewrite.
        item: RetrievalQueryItem = {
            "id": str(sample["id"]),
            "query": MultiModalInput(
                text=compose_parser_text_with_instruction(
                    "",
                    instruction=str(transform_kwargs["classification_instruction"]),
                    add_video_token=True,
                    token_separator=_classification_token_separator(transform_kwargs),
                    append_trailing_newline=_classification_append_trailing_newline(transform_kwargs),
                ),
                media=collect_video_media(sample, defaults=transform_kwargs),
            ),
        }
        metadata = extract_item_metadata(
            sample,
            consumed_fields={"id", "pos_text", "neg_text", "video", "images", "media_metadata"},
        )
        if metadata:
            item["metadata"] = metadata
        return item

    item = {
        "id": str(sample["id"]),
        "candidate": MultiModalInput(
            text=str(sample.get("text") or ""),
            media=[],
        ),
    }
    metadata = extract_item_metadata(sample, consumed_fields={"id", "text"})
    if metadata:
        item["metadata"] = metadata
    return item


def build_query_transform(*, dataset_name: str, transform_kwargs: Mapping[str, Any] | None = None) -> SampleTransform:
    _require_dataset_name(dataset_name)
    defaults = _normalize_transform_kwargs(transform_kwargs)
    return partial(
        _transform_ssv2_item,
        role="query",
        transform_kwargs=defaults,
    )


def build_candidate_transform(*, dataset_name: str, transform_kwargs: Mapping[str, Any] | None = None) -> SampleTransform:
    _require_dataset_name(dataset_name)
    defaults = _normalize_transform_kwargs(transform_kwargs)
    return partial(
        _transform_ssv2_item,
        role="candidate",
        transform_kwargs=defaults,
    )


__all__ = [
    "DATASET_NAME",
    "QUERY_READ_COLUMNS",
    "CANDIDATE_READ_COLUMNS",
    "DEFAULT_TRANSFORM_KWARGS",
    "get_query_read_columns",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "build_query_transform",
    "build_candidate_transform",
]
