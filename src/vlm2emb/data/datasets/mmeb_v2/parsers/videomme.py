"""MMEB parser for Video-MME.

The archive query shape keeps the question and option list inline after the
video token. Canonical keeps the same wording and option block, but the parser
reformats the query so the standalone video token occupies its own line and the
complete prompt ends with one trailing newline.
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
    require_non_empty_string,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

DATASET_NAME = "Video-MME"
PATH_REL = "video-tasks/Video-MME"
MODALITY = "video"
TASK_TYPE = "video_question_answer"
QUERY_READ_COLUMNS = ("id", "question", "options", "video")
CANDIDATE_READ_COLUMNS = ("id", "text")
# Video-MME owns its runtime defaults; they should not be inherited from any
# other video QA parser just because the values currently look similar. The
# parser also owns the canonical query surface directly so loader-level text
# rewrites are never required to recover token-newline or trailing-newline
# behavior.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "video_query_prefix": "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: ",
    "video_candidate_prefix": "Understand the content of the provided video.",
    "video_query_token_separator_origin": " ",
    "video_query_token_separator_canonical": "\n",
    "video_query_append_trailing_newline_origin": False,
    "video_query_append_trailing_newline_canonical": True,
    # Keep the cleanup implementation available for focused experiments, but
    # default both origin and canonical to the raw archive answer surface.
    "choice_prefix_cleanup": False,
}

def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported Video-MME parser dataset: {dataset_name}")


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
    # Canonical mode only normalizes question whitespace; origin preserves the
    # archive punctuation and line-wrapping exactly. Options stay untouched in
    # both modes.
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


def _candidate_media(sample: dict[str, Any], *, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    return collect_video_media(sample, defaults=defaults, frame_key="num_frames")


def _strip_choice_prefix(text: str) -> str:
    """Drop answer labels when conversion already included them."""
    stripped = text.strip()
    for prefix in ("(A) ", "(B) ", "(C) ", "(D) ", "(E) "):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :]
    if len(stripped) >= 3 and stripped[1:3] == ". ":
        return stripped[3:]
    if len(stripped) >= 3 and stripped[1:3] == ") ":
        return stripped[3:]
    return stripped


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
    options = sample.get("options") or []
    normalized_options = [str(value) for value in options if value is not None]
    # Archive keeps the option list on separate lines after the question.
    query_text = question + "\n" + "\n".join(normalized_options)
    composed = process_input_text(
        _video_default(defaults, "video_query_prefix", DEFAULT_TRANSFORM_KWARGS["video_query_prefix"]),
        text=query_text,
        add_video_token=True,
        token_separator=_query_token_separator(defaults),
    )
    # Canonical keeps archive wording and internal option structure, but the
    # parser now owns the token-on-own-line layout plus the final newline so
    # this query surface no longer depends on any loader-level rewrite.
    if _query_append_trailing_newline(defaults):
        composed = composed + "\n"
    return composed


def _compose_candidate_text(sample: dict[str, Any]) -> str:
    text = str(sample.get("text") or "")
    prefix_index = text.find(". ")
    if prefix_index >= 0:
        return text[prefix_index + 2 :].strip(". ")
    return text


def _candidate_text(sample: dict[str, Any], *, defaults: dict[str, Any]) -> str:
    raw_text = str(sample.get("text") or "")
    if bool(defaults.get("choice_prefix_cleanup", False)):
        # The helper is intentionally retained even though current origin and
        # canonical defaults keep it disabled. Some audits still need a way to
        # compare cleaned answer-only surfaces against archive-prefixed data.
        return _strip_choice_prefix(_compose_candidate_text(sample))
    return raw_text


def _video_default(defaults: dict[str, Any], key: str, fallback: str) -> str:
    value = defaults.get(key)
    return fallback if value is None else str(value)


def _build_videomme_item(
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
        consumed_fields = {"id", "question", "options", "video", "images", "image", "media_metadata"}
    else:
        item = {
            "id": str(sample["id"]),
            "candidate": MultiModalInput(
                text=_candidate_text(sample, defaults=defaults),
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
    return partial(_build_videomme_item, role="query", defaults=normalized)


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized = _normalize_transform_kwargs(transform_kwargs)
    return partial(_build_videomme_item, role="candidate", defaults=normalized)


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
