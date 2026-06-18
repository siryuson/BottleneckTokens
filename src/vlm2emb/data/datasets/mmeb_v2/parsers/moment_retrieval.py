"""MMEB parser for moment retrieval datasets.

Origin/canonical contract owned by this parser:
- origin query/candidate keep archive-compatible single-line visual-token surface;
- canonical query/candidate own final runtime surface directly:
  standalone visual token line + exactly one terminal newline for full prompt.

Why newline rules live here (instead of loader):
- this parser owns prompt wording and whether the token is standalone or embedded;
- only parser-level logic can preserve archive wording while applying canonical
  layout consistently across all samples in this dataset family.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.const import normalize_text_whitespace
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_video_media,
    extract_item_metadata,
    compose_parser_text_with_instruction,
    require_non_empty_string,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "QVHighlight": ("id", "query", "video"),
    "Charades-STA": ("id", "query", "video"),
}
DATASET_NAMES = tuple(QUERY_READ_COLUMNS_BY_DATASET)
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "QVHighlight": ("id", "video"),
    "Charades-STA": ("id", "video"),
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    "QVHighlight": {
        "runtime_mode": "canonical",
        "num_frames": 8,
        "moment_query_prefix": "Find the clip that corresponds to the described scene in the given video:",
        "moment_candidate_prefix": "Understand the content of the provided video.",
    },
    "Charades-STA": {
        "runtime_mode": "canonical",
        "num_frames": 8,
        "moment_query_prefix": "Find the clip that corresponds to the described scene in the given video:",
        "moment_candidate_prefix": "Understand the content of the provided video.",
    },
}


def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unsupported moment retrieval parser dataset: {dataset_name}")


def _normalize_transform_kwargs(
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return QUERY_READ_COLUMNS_BY_DATASET[dataset_name]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return CANDIDATE_READ_COLUMNS_BY_DATASET[dataset_name]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    _require_dataset_name(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])


def _runtime_mode(defaults: dict[str, Any]) -> str:
    return str(defaults.get("runtime_mode") or "canonical")


def _normalize_for_mode(text: str, *, mode: str) -> str:
    # Canonical mode normalizes spacing while deferring terminal-newline
    # ownership to compose_parser_text_with_instruction(..., append_trailing_newline).
    # Origin keeps archive spacing unchanged.
    if mode == "canonical":
        return normalize_text_whitespace(text).rstrip("\n")
    return text


def _query_media(sample: dict[str, Any], *, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    return collect_video_media(sample, defaults=defaults, frame_key="num_frames")


def _candidate_media(sample: dict[str, Any], *, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    return collect_video_media(sample, defaults=defaults, frame_key="num_frames")


def _compose_query_text(dataset_name: str, sample: dict[str, Any], *, defaults: dict[str, Any]) -> str:
    mode = _runtime_mode(defaults)
    canonical = mode == "canonical"
    query = require_non_empty_string(
        sample,
        "query",
        dataset_name=dataset_name,
        task_type="video_moment_retrieval",
        role="query",
    )
    query = _normalize_for_mode(query, mode=mode)
    return compose_parser_text_with_instruction(
        query,
        instruction=str(
            defaults.get("moment_query_prefix")
            or DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name]["moment_query_prefix"]
        ),
        add_video_token=True,
        # Query contract:
        # - origin: inline visual token ("<|video_pad|> instruction: body")
        # - canonical: visual token on its own line + single terminal newline.
        token_separator="\n" if canonical else " ",
        append_trailing_newline=canonical,
    )


def _compose_candidate_text(defaults: dict[str, Any]) -> str:
    canonical = _runtime_mode(defaults) == "canonical"
    return compose_parser_text_with_instruction(
        "",
        instruction=str(defaults["moment_candidate_prefix"]),
        add_video_token=True,
        # Candidate contract:
        # - origin: keep archive single-line candidate prompt;
        # - canonical: switch to parser-owned block layout + terminal newline.
        # Raw-table audit confirmed one fixed candidate shape (media-only plus
        # parser-owned instruction), so this canonical upgrade is safe.
        token_separator="\n" if canonical else " ",
        append_trailing_newline=canonical,
    )


def _build_moment_retrieval_item(
    dataset_name: str,
    sample: dict[str, Any],
    *,
    role: str,
    defaults: dict[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    if role not in {"query", "candidate"}:
        raise ValueError(f"Unsupported MMEB video eval role: {role}")

    if role == "query":
        item: RetrievalQueryItem = {
            "id": str(sample["id"]),
            "query": MultiModalInput(
                text=_compose_query_text(dataset_name, sample, defaults=defaults),
                media=_query_media(sample, defaults=defaults),
            ),
        }
        consumed_fields = {"id", "query", "video", "images", "image", "media_metadata"}
    else:
        item = {
            "id": str(sample["id"]),
            "candidate": MultiModalInput(
                text=_compose_candidate_text(defaults),
                media=_candidate_media(sample, defaults=defaults),
            ),
        }
        consumed_fields = {"id", "video", "images", "image", "media_metadata"}

    metadata = extract_item_metadata(sample, consumed_fields=consumed_fields)
    if metadata:
        item["metadata"] = metadata
    return item


def build_query_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized = _normalize_transform_kwargs(dataset_name, transform_kwargs)
    return partial(_build_moment_retrieval_item, dataset_name, role="query", defaults=normalized)


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized = _normalize_transform_kwargs(dataset_name, transform_kwargs)
    return partial(_build_moment_retrieval_item, dataset_name, role="candidate", defaults=normalized)


__all__ = [
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "QUERY_READ_COLUMNS_BY_DATASET",
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DATASET_NAMES",
    "get_query_read_columns",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "build_query_transform",
    "build_candidate_transform",
]
