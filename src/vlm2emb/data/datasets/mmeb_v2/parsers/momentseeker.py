"""MMEB parser for MomentSeeker.

MomentSeeker has three query modalities with parser-owned contracts:
- text-only query: no visual token, no media;
- image-conditioned query: image token + image media;
- video-conditioned query: video token + video media.

Origin/canonical split is explicit and parser-owned:
- origin keeps archive-compatible single-line token layout;
- canonical owns final runtime surface:
  standalone visual token line + exactly one terminal newline.

Why this parser owns newline/token layout:
- branch selection (text/image/video) is sample-level and parser-specific;
- loader/processor cannot safely infer these boundaries without losing
  archive wording parity or modality intent.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.const import normalize_text_whitespace
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_video_media,
    compose_parser_text_with_instruction,
    extract_item_metadata,
    require_non_empty_string,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

DATASET_NAME = "MomentSeeker"
PATH_REL = "video-tasks/MomentSeeker"
MODALITY = "video"
TASK_TYPE = "video_moment_retrieval"
QUERY_READ_COLUMNS = ("id", "query", "image", "video")
CANDIDATE_READ_COLUMNS = ("id", "video")
# MomentSeeker owns its runtime defaults. The query-side prompts are split by
# input modality to match the archive parser.
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    "num_frames": 8,
    "momentseeker_text_query_prefix": "Find the clip that corresponds to the given text:",
    "momentseeker_image_query_prefix": "Select the video clip that aligns with the given text and image:",
    "momentseeker_video_query_prefix": "Find the clip that corresponds to the given sentence and video segment:",
    "moment_candidate_prefix": "Understand the content of the provided video clip.",
}

def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported MomentSeeker parser dataset: {dataset_name}")


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
    # Canonical normalizes spacing but defers final terminal-newline ownership
    # to compose_parser_text_with_instruction(..., append_trailing_newline).
    # Origin preserves archive line wrapping/spacing.
    if mode == "canonical":
        return normalize_text_whitespace(text).rstrip("\n")
    return text


def _query_media(sample: dict[str, Any], *, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    # Query modality boundary:
    # - image present and no video -> image-conditioned query;
    # - video present -> video-conditioned query;
    # - neither image nor video -> text-only query.
    if sample.get("image") is not None and not sample.get("video"):
        return [{"kind": "image", "content": sample["image"]}]
    if sample.get("video"):
        return collect_video_media(sample, defaults=defaults, frame_key="num_frames")
    return []


def _candidate_media(sample: dict[str, Any], *, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    return collect_video_media(sample, defaults=defaults, frame_key="num_frames")


def _compose_query_text(sample: dict[str, Any], *, defaults: dict[str, Any]) -> str:
    mode = _runtime_mode(defaults)
    canonical = mode == "canonical"
    query = require_non_empty_string(
        sample,
        "query",
        dataset_name=DATASET_NAME,
        task_type="video_moment_retrieval",
        role="query",
    )
    query = _normalize_for_mode(query, mode=mode)
    if sample.get("image") is not None and not sample.get("video"):
        return compose_parser_text_with_instruction(
            query,
            instruction=str(
                defaults.get("momentseeker_image_query_prefix")
                or DEFAULT_TRANSFORM_KWARGS["momentseeker_image_query_prefix"]
            ),
            add_image_token=True,
            # Image-conditioned contract:
            # - origin: inline image token;
            # - canonical: standalone token line + single terminal newline.
            token_separator="\n" if canonical else " ",
            append_trailing_newline=canonical,
        )
    if sample.get("video"):
        return compose_parser_text_with_instruction(
            query,
            instruction=str(
                defaults.get("momentseeker_video_query_prefix")
                or DEFAULT_TRANSFORM_KWARGS["momentseeker_video_query_prefix"]
            ),
            add_video_token=True,
            # Video-conditioned contract:
            # - origin: inline video token;
            # - canonical: standalone token line + single terminal newline.
            token_separator="\n" if canonical else " ",
            append_trailing_newline=canonical,
        )
    # Text-only contract:
    # - origin: archive wording with no forced terminal newline;
    # - canonical: keep wording but ensure one terminal newline.
    return compose_parser_text_with_instruction(
        query,
        instruction=str(
            defaults.get("momentseeker_text_query_prefix")
            or DEFAULT_TRANSFORM_KWARGS["momentseeker_text_query_prefix"]
        ),
        append_trailing_newline=canonical,
    )


def _compose_candidate_text(defaults: dict[str, Any]) -> str:
    canonical = _runtime_mode(defaults) == "canonical"
    return compose_parser_text_with_instruction(
        "",
        instruction=str(defaults.get("moment_candidate_prefix") or DEFAULT_TRANSFORM_KWARGS["moment_candidate_prefix"]),
        add_video_token=True,
        # Candidate contract:
        # - origin: archive single-line visual-token prompt;
        # - canonical: parser-owned block layout + single terminal newline.
        # Full raw-table audit confirmed one uniform candidate shape:
        # video media plus fixed parser-owned instruction.
        token_separator="\n" if canonical else " ",
        append_trailing_newline=canonical,
    )


def _build_momentseeker_item(
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
        consumed_fields = {"id", "query", "image", "video", "images", "media_metadata", "text"}
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
    normalized = _normalize_transform_kwargs(transform_kwargs)
    return partial(_build_momentseeker_item, role="query", defaults=normalized)


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    normalized = _normalize_transform_kwargs(transform_kwargs)
    return partial(_build_momentseeker_item, role="candidate", defaults=normalized)


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
