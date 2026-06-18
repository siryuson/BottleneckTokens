from __future__ import annotations

"""MMEB parser for MSR-VTT video retrieval.

The parser keeps the archive prompt wording and dataset-specific defaults local
to the file. Origin matches the archive surface form; canonical only applies
the parser-local whitespace normalization and newline policy defined here.

Surface contract summary:
- Query: origin keeps the archive tail state; canonical enforces one trailing newline.
- Candidate: origin keeps inline video-token wording; canonical uses newline token
  separation plus one trailing newline.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.const import normalize_text_whitespace
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_video_media,
    extract_item_metadata,
    compose_parser_text_with_instruction,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

DATASET_NAME = "MSR-VTT"
PATH_REL = "video-tasks/MSR-VTT"
MODALITY = "video"
TASK_TYPE = "video_retrieval"
# The parser-local read contract is kept explicit so query/candidate coverage
# remains reviewable without chasing external configuration.
QUERY_READ_COLUMNS = ("id", "caption")
CANDIDATE_READ_COLUMNS = ("id", "video_id", "video")
# Prompt defaults are dataset-local because the archive wording is part of the
# runtime contract, not a shared family default.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "video_query_prefix": "Find a video that contains the following visual content:",
    "video_candidate_prefix": "Understand the content of the provided video.",
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
    # Caller overrides are applied on top of the dataset-local defaults.
    normalized = dict(DEFAULT_TRANSFORM_KWARGS)
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def _normalize_query_text(raw_text: Any, *, runtime_mode: str) -> str:
    if isinstance(raw_text, list):
        raw_text = next(
            (str(value) for value in raw_text if isinstance(value, str) and value.strip()),
            "",
        )
    text = str(raw_text or "")
    if runtime_mode == "canonical":
        return normalize_text_whitespace(text).rstrip("\n")
    return text


def _compose_query_text(sample: dict[str, Any], *, transform_kwargs: dict[str, Any]) -> str:
    canonical = str(transform_kwargs["runtime_mode"]) == "canonical"
    query_text = _normalize_query_text(sample.get("caption"), runtime_mode=str(transform_kwargs["runtime_mode"]))
    return compose_parser_text_with_instruction(
        query_text,
        instruction=str(transform_kwargs["video_query_prefix"]),
        # Canonical query newline is owned by this parser so retrieval prompts
        # stay stable after the temporary loader-level experiment is removed.
        append_trailing_newline=canonical,
    )


def _compose_candidate_text(*, transform_kwargs: dict[str, Any]) -> str:
    canonical = str(transform_kwargs["runtime_mode"]) == "canonical"
    return compose_parser_text_with_instruction(
        "",
        instruction=str(transform_kwargs["video_candidate_prefix"]),
        add_video_token=True,
        # Canonical candidate layout is also parser-owned: block style with one
        # trailing newline for canonical, single-line archive wording for origin.
        token_separator="\n" if canonical else " ",
        append_trailing_newline=canonical,
    )


def _transform_query_item(
    sample: dict[str, Any],
    *,
    transform_kwargs: dict[str, Any],
) -> RetrievalQueryItem:
    return {
        "id": str(sample["id"]),
        "query": MultiModalInput(
            text=_compose_query_text(sample, transform_kwargs=transform_kwargs),
            media=[],
        ),
    }


def _transform_candidate_item(
    sample: dict[str, Any],
    *,
    transform_kwargs: dict[str, Any],
) -> RetrievalCandidateItem:
    media = collect_video_media(sample, defaults=transform_kwargs, frame_key="num_frames")
    if not media:
        raise ValueError("Missing required media for MMEB eval transform: dataset_name=MSR-VTT, task_type=video_retrieval, role=candidate")
    item: RetrievalCandidateItem = {
        "id": str(sample["id"]),
        "candidate": MultiModalInput(
            text=_compose_candidate_text(transform_kwargs=transform_kwargs),
            media=media,
        ),
    }
    metadata = extract_item_metadata(
        sample,
        consumed_fields={"id", "text", "images", "video", "image", "media_metadata", "caption", "video_path"},
    )
    if metadata:
        item["metadata"] = metadata
    return item


def build_query_transform(
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    defaults = _normalize_transform_kwargs(transform_kwargs)
    _require_dataset_name(dataset_name)
    return partial(_transform_query_item, transform_kwargs=defaults)


def build_candidate_transform(
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    defaults = _normalize_transform_kwargs(transform_kwargs)
    _require_dataset_name(dataset_name)
    return partial(_transform_candidate_item, transform_kwargs=defaults)


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
