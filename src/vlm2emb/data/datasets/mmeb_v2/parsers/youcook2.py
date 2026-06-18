from __future__ import annotations

"""MMEB parser for YouCook2 video retrieval.

Origin reproduces the archive recipe-summary prompt shape. Canonical keeps the
same dataset-local defaults and only applies the local normalization defined in
this parser.

Surface contract summary:
- Query: origin keeps archive tail state; canonical enforces one trailing newline.
- Candidate: origin keeps inline video-token wording; canonical emits newline
  token separation with one trailing newline.
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

DATASET_NAME = "YouCook2"
PATH_REL = "video-tasks/YouCook2"
MODALITY = "video"
TASK_TYPE = "video_retrieval"
# Dataset-local read columns keep the query/candidate contract explicit.
QUERY_READ_COLUMNS = ("id", "sentence")
CANDIDATE_READ_COLUMNS = ("id", "video", "source_id", "segment", "video_path", "youtube_id")
# Prompt defaults stay with the parser so archive wording is easy to audit.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "video_query_prefix": "Find a video that demonstrates the following action while making a recipe:",
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
    # Dataset-local defaults are the base; caller overrides stay additive.
    normalized = dict(DEFAULT_TRANSFORM_KWARGS)
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def _normalize_query_text(raw_text: Any, *, runtime_mode: str) -> str:
    text = str(raw_text or "")
    if runtime_mode == "canonical":
        return normalize_text_whitespace(text).rstrip("\n")
    return text


def _compose_query_text(sample: dict[str, Any], *, transform_kwargs: dict[str, Any]) -> str:
    canonical = str(transform_kwargs["runtime_mode"]) == "canonical"
    query_text = _normalize_query_text(sample.get("sentence"), runtime_mode=str(transform_kwargs["runtime_mode"]))
    return compose_parser_text_with_instruction(
        query_text,
        instruction=str(transform_kwargs["video_query_prefix"]),
        # Canonical query newline stays in parser-local logic so the final
        # retrieval surface does not rely on loader-level mutation.
        append_trailing_newline=canonical,
    )


def _compose_candidate_text(*, transform_kwargs: dict[str, Any]) -> str:
    canonical = str(transform_kwargs["runtime_mode"]) == "canonical"
    return compose_parser_text_with_instruction(
        "",
        instruction=str(transform_kwargs["video_candidate_prefix"]),
        add_video_token=True,
        # Canonical candidate block formatting is parser-owned for the same
        # reason: loader newline experiments should not define final prompts.
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
        raise ValueError("Missing required media for MMEB eval transform: dataset_name=YouCook2, task_type=video_retrieval, role=candidate")
    item: RetrievalCandidateItem = {
        "id": str(sample["id"]),
        "candidate": MultiModalInput(
            text=_compose_candidate_text(transform_kwargs=transform_kwargs),
            media=media,
        ),
    }
    metadata = extract_item_metadata(
        sample,
        consumed_fields={"id", "text", "images", "video", "image", "media_metadata", "sentence", "video_path"},
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
