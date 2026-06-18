"""MMEB parser for MVBench.

MVBench keeps the archive multiple-choice question format:
- query text appends raw options on separate lines;
- candidate text is answer-only in the archive surface form, even if the
  converted artifact stores an option label.
Canonical keeps the same wording and choice block, but parser-owned canonical
formatting moves the video token onto its own line and ends the full prompt
with one trailing newline.
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
    process_input_text,
    format_choice_template,
    require_non_empty_string,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

DATASET_NAME = "MVBench"
PATH_REL = "video-tasks/MVBench"
MODALITY = "video"
TASK_TYPE = "video_question_answer"
QUERY_READ_COLUMNS = ("id", "question", "candidates", "video")
CANDIDATE_READ_COLUMNS = ("id", "text")
# MVBench owns this runtime default bundle. Canonical surface differences stay
# local to the parser so the global MMEB loader never needs to reinterpret a
# choice prompt after formatting.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "video_query_prefix": "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: ",
    "video_query_token_separator_origin": " ",
    "video_query_token_separator_canonical": "\n",
    "video_query_append_trailing_newline_origin": False,
    "video_query_append_trailing_newline_canonical": True,
}

def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported MVBench parser dataset: {dataset_name}")


def _normalize_transform_kwargs(transform_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = dict(DEFAULT_TRANSFORM_KWARGS)
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return QUERY_READ_COLUMNS


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return CANDIDATE_READ_COLUMNS


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    _require_dataset_name(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS)


def _runtime_mode(defaults: dict[str, Any]) -> str:
    return str(defaults.get("runtime_mode") or "canonical")


def _normalize_for_mode(text: str, *, mode: str) -> str:
    # Canonical owns whitespace cleanup for MVBench queries. Origin preserves
    # archive spacing, including any trailing spaces left by conversion.
    if mode == "canonical":
        return normalize_text_whitespace(text).rstrip("\n")
    return text


def _normalize_candidate_for_mode(text: str, *, mode: str) -> str:
    # Canonical also normalizes candidate whitespace so trailing spaces from
    # rerun artifacts stop leaking into the runtime answer surface. The option
    # label itself remains part of the parser-owned wording.
    if mode == "canonical":
        return normalize_text_whitespace(text).rstrip("\n")
    return text


def _query_token_separator(defaults: dict[str, Any]) -> str:
    explicit = defaults.get("video_query_token_separator")
    if explicit is not None:
        return str(explicit)
    return str(defaults.get(f"video_query_token_separator_{_runtime_mode(defaults)}", " "))


def _query_append_trailing_newline(defaults: dict[str, Any]) -> bool:
    explicit = defaults.get("video_query_append_trailing_newline")
    if explicit is not None:
        return bool(explicit)
    return bool(defaults.get(f"video_query_append_trailing_newline_{_runtime_mode(defaults)}", False))


def _query_media(sample: dict[str, Any], *, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    return collect_video_media(sample, defaults=defaults, frame_key="num_frames")


def _compose_query_text(sample: dict[str, Any], *, defaults: dict[str, Any]) -> str:
    mode = _runtime_mode(defaults)
    question = require_non_empty_string(
        sample,
        "question",
        dataset_name=DATASET_NAME,
        task_type="video_question_answer",
        role="query",
    )
    question = _normalize_for_mode(question, mode=mode)
    options = [
        _normalize_for_mode(str(value), mode=mode)
        for value in (sample.get("candidates") or [])
        if value is not None
    ]
    composed = process_input_text(
        _video_default(defaults, "video_query_prefix", DEFAULT_TRANSFORM_KWARGS["video_query_prefix"]),
        # Archive formats MVBench as a standard labeled choice template.
        text=format_choice_template(question, options),
        add_video_token=True,
        token_separator=_query_token_separator(defaults),
    )
    # Canonical keeps the archive choice wording, including internal option
    # line breaks, but the parser owns whether the standalone video token stays
    # inline or moves to its own line and whether the full block ends with one
    # newline. Keeping that contract here avoids loader-side prompt surgery.
    if _query_append_trailing_newline(defaults):
        composed = composed + "\n"
    return composed


def _video_default(defaults: dict[str, Any], key: str, fallback: str) -> str:
    value = defaults.get(key)
    return fallback if value is None else str(value)


def _build_mvbench_item(
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
                text=_compose_query_text(sample, defaults=defaults),
                media=_query_media(sample, defaults=defaults),
            ),
        }
        consumed_fields = {
            "id",
            "question",
            "candidates",
            "video",
            "images",
            "image",
            "media_metadata",
        }
    else:
        mode = _runtime_mode(defaults)
        item = {
            "id": str(sample["id"]),
            "candidate": MultiModalInput(
                # MVBench keeps the prefixed candidate surface in the archive
                # and the converted artifact. Canonical only removes whitespace
                # artifacts such as trailing spaces; it does not rewrite labels.
                text=_normalize_candidate_for_mode(str(sample.get("text") or ""), mode=mode),
                media=[],
            ),
        }
        consumed_fields = {"id", "text"}

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
    normalized = _normalize_transform_kwargs(transform_kwargs)
    return partial(_build_mvbench_item, role="query", defaults=normalized)


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized = _normalize_transform_kwargs(transform_kwargs)
    return partial(_build_mvbench_item, role="candidate", defaults=normalized)


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
