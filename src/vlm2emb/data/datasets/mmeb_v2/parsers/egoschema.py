"""MMEB parser for EgoSchema.

This parser keeps the archive's compact option concatenation:
- query text appends the option list directly after the question;
- candidate text keeps the archive answer surface by default.

Canonical keeps the same wording, but the parser now owns the standalone
video-token line plus the single trailing newline for the completed prompt.
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

DATASET_NAME = "EgoSchema"
PATH_REL = "video-tasks/EgoSchema"
MODALITY = "video"
TASK_TYPE = "video_question_answer"
QUERY_READ_COLUMNS = ("id", "question", "option", "video")
CANDIDATE_READ_COLUMNS = ("id", "text")
# EgoSchema owns these runtime defaults. Parser-owned canonical formatting keeps
# the archive wording intact while avoiding any loader-level text rewrite.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "video_query_prefix": "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: ",
    "video_query_token_separator_origin": " ",
    "video_query_token_separator_canonical": "\n",
    "video_query_append_trailing_newline_origin": False,
    "video_query_append_trailing_newline_canonical": True,
    # Keep the helper for audits, but disable the cleanup rule by default in
    # both origin and canonical so runtime matches the archive answer surface.
    "choice_prefix_cleanup": False,
}

def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported EgoSchema parser dataset: {dataset_name}")


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
    # Canonical mode only collapses whitespace; origin preserves the archive
    # prompt surface.
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


def _strip_archive_choice_prefix(text: str) -> str:
    prefix_index = text.find(". ")
    if prefix_index >= 0:
        return text[prefix_index + 2 :].strip(". ")
    return text


def _candidate_text(sample: dict[str, Any], *, defaults: dict[str, Any]) -> str:
    raw_text = str(sample.get("text") or "")
    if bool(defaults.get("choice_prefix_cleanup", False)):
        # The cleanup helper remains available for targeted experiments, but it
        # is not part of the default origin/canonical contract anymore.
        return _strip_archive_choice_prefix(raw_text)
    return raw_text


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
    options = sample.get("option") or []
    if isinstance(options, list):
        options_text = " ".join(str(value) for value in options if value is not None)
    else:
        options_text = str(options)
    # Archive concatenates the choices directly after the question.
    composed = process_input_text(
        _video_default(defaults, "video_query_prefix", DEFAULT_TRANSFORM_KWARGS["video_query_prefix"]),
        text=question + options_text,
        add_video_token=True,
        token_separator=_query_token_separator(defaults),
    )
    # Canonical keeps the compact archive wording untouched, but the parser
    # explicitly owns the newline-separated token layout and single trailing
    # newline so this behavior does not depend on a loader patch.
    if _query_append_trailing_newline(defaults):
        composed = composed + "\n"
    return composed


def _video_default(defaults: dict[str, Any], key: str, fallback: str) -> str:
    value = defaults.get(key)
    return fallback if value is None else str(value)


def _build_egoschema_item(
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
            "option",
            "video",
            "images",
            "image",
            "media_metadata",
        }
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
    return partial(_build_egoschema_item, role="query", defaults=normalized)


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized = _normalize_transform_kwargs(transform_kwargs)
    return partial(_build_egoschema_item, role="candidate", defaults=normalized)


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
