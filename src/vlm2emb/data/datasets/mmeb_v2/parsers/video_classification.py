from __future__ import annotations

"""MMEB video classification parsers for K700, UCF101, HMDB51, and Breakfast.

Each dataset keeps its own read-column contract and prompt defaults so the
origin path stays aligned with the archive wording. Canonical intentionally
changes only parser-owned surface formatting: the visual token moves onto its
own line and the complete prompt ends with one trailing newline.
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

# These columns define the dataset-local read contract for each video
# classification benchmark. Keep them close to the parser so prompt and input
# assumptions stay reviewable together.
QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "K700": ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"),
    "UCF101": ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"),
    "HMDB51": ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"),
    "Breakfast": ("id", "qry_instruction", "qry_text", "video", "video_id", "video_path"),
}
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "K700": ("id", "text"),
    "UCF101": ("id", "text"),
    "HMDB51": ("id", "text"),
    "Breakfast": ("id", "text"),
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    "K700": {
        "runtime_mode": "canonical",
        "num_frames": 8,
        "classification_instruction": "Recognize the category of the video content.",
        "classification_token_separator_origin": " ",
        "classification_token_separator_canonical": "\n",
        # Canonical keeps the archive wording but promotes the loader-era
        # trailing newline plus token-on-own-line layout into parser-owned
        # formatting so query shape no longer depends on the global MMEB loader
        # patch.
        "classification_append_trailing_newline_origin": False,
        "classification_append_trailing_newline_canonical": True,
    },
    "UCF101": {
        "runtime_mode": "canonical",
        "num_frames": 8,
        "classification_instruction": "What activities or sports are being performed by the person in the video?",
        "classification_token_separator_origin": " ",
        "classification_token_separator_canonical": "\n",
        # Origin follows the archive parser prompt exactly; canonical keeps the
        # wording but moves the visual token onto its own line and appends the
        # parser-owned trailing newline that used to come from the loader.
        "classification_append_trailing_newline_origin": False,
        "classification_append_trailing_newline_canonical": True,
    },
    "HMDB51": {
        "runtime_mode": "canonical",
        "num_frames": 8,
        "classification_instruction": "What actions or objects interactions are the person in the video doing?",
        "classification_token_separator_origin": " ",
        "classification_token_separator_canonical": "\n",
        # Keep the archive parser wording unchanged in origin. Canonical only
        # changes parser-owned surface formatting by making the video token a
        # standalone line and restoring one parser-local trailing newline.
        "classification_append_trailing_newline_origin": False,
        "classification_append_trailing_newline_canonical": True,
    },
    "Breakfast": {
        "runtime_mode": "canonical",
        "num_frames": 8,
        # Keep the trailing space to mirror the archive Breakfast instruction verbatim.
        "classification_instruction": "Recognize the breakfast type that the person is cooking in the video. ",
        "classification_token_separator_origin": " ",
        "classification_token_separator_canonical": "\n",
        # Breakfast keeps the archive parser wording in origin, including the
        # trailing formatting artifact. Canonical trims that artifact, moves
        # the visual token to its own line, and then appends one parser-owned
        # trailing newline.
        "classification_append_trailing_newline_origin": False,
        "classification_append_trailing_newline_canonical": True,
    },
}

DATASET_NAMES = tuple(QUERY_READ_COLUMNS_BY_DATASET)


def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name not in QUERY_READ_COLUMNS_BY_DATASET:
        raise ValueError(f"Unsupported parser dataset: {dataset_name}")


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return QUERY_READ_COLUMNS_BY_DATASET[dataset_name]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    _require_dataset_name(dataset_name)
    return CANDIDATE_READ_COLUMNS_BY_DATASET[dataset_name]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    _require_dataset_name(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])


def _normalize_transform_kwargs(dataset_name: str, transform_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # Caller overrides are merged on top of the dataset-local defaults.
    normalized = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def _runtime_mode(transform_kwargs: Mapping[str, Any]) -> str:
    return str(transform_kwargs.get("runtime_mode") or "canonical")


def _classification_append_trailing_newline(transform_kwargs: Mapping[str, Any]) -> bool:
    # Keep the legacy override key working so parser callers can still force one
    # behavior explicitly while the default stays mode-specific.
    explicit = transform_kwargs.get("classification_append_trailing_newline")
    if explicit is not None:
        return bool(explicit)
    mode = _runtime_mode(transform_kwargs)
    key = f"classification_append_trailing_newline_{mode}"
    return bool(transform_kwargs.get(key, False))


def _classification_token_separator(transform_kwargs: Mapping[str, Any]) -> str:
    explicit = transform_kwargs.get("classification_token_separator")
    if explicit is not None:
        return str(explicit)
    mode = _runtime_mode(transform_kwargs)
    key = f"classification_token_separator_{mode}"
    return str(transform_kwargs.get(key, " "))


def _transform_video_classification_item(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    role: str,
    transform_kwargs: dict[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    _require_dataset_name(dataset_name)
    if role == "query":
        # Classification prompts are fully dataset-local. Breakfast is the only
        # subset in this group whose archive parser keeps a trailing space at
        # the end of the instruction; canonical trims that terminal artifact
        # but keeps the wording itself unchanged. The origin/canonical layout
        # split lives here instead of the loader so each parser can explicitly
        # own whether its video token stays inline or becomes a standalone line.
        instruction = str(transform_kwargs["classification_instruction"])
        if dataset_name == "Breakfast" and _runtime_mode(transform_kwargs) == "canonical":
            instruction = instruction.rstrip()
        item: RetrievalQueryItem = {
            "id": str(sample["id"]),
            "query": MultiModalInput(
                text=compose_parser_text_with_instruction(
                    "",
                    instruction=instruction,
                    add_video_token=True,
                    token_separator=_classification_token_separator(transform_kwargs),
                    append_trailing_newline=_classification_append_trailing_newline(transform_kwargs),
                ),
                media=collect_video_media(sample, defaults=transform_kwargs),
            ),
        }
        metadata = extract_item_metadata(
            sample,
            consumed_fields={"id", "qry_instruction", "qry_text", "video", "images", "media_metadata"},
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
    defaults = _normalize_transform_kwargs(dataset_name, transform_kwargs)
    return partial(
        _transform_video_classification_item,
        dataset_name=dataset_name,
        role="query",
        transform_kwargs=defaults,
    )


def build_candidate_transform(*, dataset_name: str, transform_kwargs: Mapping[str, Any] | None = None) -> SampleTransform:
    defaults = _normalize_transform_kwargs(dataset_name, transform_kwargs)
    return partial(
        _transform_video_classification_item,
        dataset_name=dataset_name,
        role="candidate",
        transform_kwargs=defaults,
    )


__all__ = [
    "QUERY_READ_COLUMNS_BY_DATASET",
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "DATASET_NAMES",
    "get_query_read_columns",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "build_query_transform",
    "build_candidate_transform",
]
