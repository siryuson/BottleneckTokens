"""VisRAG visdoc runtime parser.

This file only owns the VisRAG dataset contract:

- the subset coverage table;
- per-dataset read columns;
- dataset-local default transform kwargs;
- thin query/candidate transform builders.

The archive-style prompt wording is preserved in `origin`, while `canonical`
only normalizes whitespace/newline shape using the same visdoc family contract:

- query keeps the same wording; canonical enforces a single trailing newline;
- origin remains non-normalizing (no forced strip/append) so source tail shape
  stays archive-aligned;
- candidate switches from `"<|image_pad|> instruction"` to the canonical
  two-line layout `"<|image_pad|>\\ninstruction\\n"`.

Conversion already writes the image payloads, so runtime does not need a
filename-hashing compatibility knob.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN, normalize_text_whitespace
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import collect_image_media, extract_item_metadata
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

_VISDOC_DEFAULTS: dict[str, Any] = {
    # Parser-local defaults are intentionally expressed with the shared rule
    # vocabulary so docs and code stay aligned without introducing a global
    # rule engine.
    #
    # Rule mapping:
    # - runtime_mode: selects the default config group (`origin` or `canonical`)
    # - canonical_normalize_query_text: whitespace_normalization for query text
    # - query_append_trailing_newline: trailing_newline for query text
    # - candidate_token_separator: visual_token_separator for candidate text
    # - candidate_append_trailing_newline: trailing_newline for candidate text
    "runtime_mode": "canonical",
    "query_instruction": "Find a document image that matches the given query:",
    "candidate_instruction": "Understand the content of the provided document image.",
    "canonical_normalize_query_text": True,
    "query_append_trailing_newline": True,
    "candidate_token_separator": "\n",
    "candidate_append_trailing_newline": True,
}

# VisRAG uses the same document-image prompt family as ViDoRe, but the subset
# split is different and the conversion artifacts already carry the final image
# payloads. Candidate `path` is still read here so it can remain visible in
# runtime metadata for subset-level debugging.
_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "VisRAG_ArxivQA": {
        "path_rel": "visdoc-tasks/VisRAG_ArxivQA",
        "modality": "visdoc",
        "task_type": "visdoc_visrag",
        "query_read_columns": ("id", "query"),
        "candidate_read_columns": ("id", "image", "path"),
        "default_transform_kwargs": dict(_VISDOC_DEFAULTS),
    },
    "VisRAG_ChartQA": {
        "path_rel": "visdoc-tasks/VisRAG_ChartQA",
        "modality": "visdoc",
        "task_type": "visdoc_visrag",
        "query_read_columns": ("id", "query"),
        "candidate_read_columns": ("id", "image", "path"),
        "default_transform_kwargs": dict(_VISDOC_DEFAULTS),
    },
    "VisRAG_MP-DocVQA": {
        "path_rel": "visdoc-tasks/VisRAG_MP-DocVQA",
        "modality": "visdoc",
        "task_type": "visdoc_visrag",
        "query_read_columns": ("id", "query"),
        "candidate_read_columns": ("id", "image", "path"),
        "default_transform_kwargs": dict(_VISDOC_DEFAULTS),
    },
    "VisRAG_SlideVQA": {
        "path_rel": "visdoc-tasks/VisRAG_SlideVQA",
        "modality": "visdoc",
        "task_type": "visdoc_visrag",
        "query_read_columns": ("id", "query"),
        "candidate_read_columns": ("id", "image", "path"),
        "default_transform_kwargs": dict(_VISDOC_DEFAULTS),
    },
    "VisRAG_InfoVQA": {
        "path_rel": "visdoc-tasks/VisRAG_InfoVQA",
        "modality": "visdoc",
        "task_type": "visdoc_visrag",
        "query_read_columns": ("id", "query"),
        "candidate_read_columns": ("id", "image", "path"),
        "default_transform_kwargs": dict(_VISDOC_DEFAULTS),
    },
    "VisRAG_PlotQA": {
        "path_rel": "visdoc-tasks/VisRAG_PlotQA",
        "modality": "visdoc",
        "task_type": "visdoc_visrag",
        "query_read_columns": ("id", "query"),
        "candidate_read_columns": ("id", "image", "path"),
        "default_transform_kwargs": dict(_VISDOC_DEFAULTS),
    },
}
DATASET_NAMES = tuple(_DATASET_SPECS)

QUERY_READ_COLUMNS_BY_DATASET = {
    name: tuple(spec["query_read_columns"])
    for name, spec in _DATASET_SPECS.items()
}
CANDIDATE_READ_COLUMNS_BY_DATASET = {
    name: tuple(spec["candidate_read_columns"])
    for name, spec in _DATASET_SPECS.items()
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET = {
    name: dict(spec["default_transform_kwargs"])
    for name, spec in _DATASET_SPECS.items()
}


def _dataset_spec(dataset_name: str) -> dict[str, Any]:
    try:
        return _DATASET_SPECS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported VisRAG parser dataset: {dataset_name}") from exc


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    _dataset_spec(dataset_name)
    return QUERY_READ_COLUMNS_BY_DATASET[dataset_name]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    _dataset_spec(dataset_name)
    return CANDIDATE_READ_COLUMNS_BY_DATASET[dataset_name]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    _dataset_spec(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])


def _normalize_transform_kwargs(
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Defaults are copied per dataset so an override on one VisRAG subset never
    # leaks into the others.
    normalized = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized.update(dict(transform_kwargs))
    return normalized


def _visdoc_default(defaults: dict[str, Any], key: str, fallback: str) -> str:
    value = defaults.get(key)
    return fallback if value is None else str(value)


def _visdoc_bool_default(defaults: dict[str, Any], key: str, fallback: bool) -> bool:
    value = defaults.get(key)
    return fallback if value is None else bool(value)


def _compose_query_text(query: str, *, defaults: dict[str, Any], mode: str) -> str:
    instruction = _visdoc_default(defaults, "query_instruction", _VISDOC_DEFAULTS["query_instruction"])
    if mode == "origin":
        # Keep archive parser surface: single-line "instruction + query"
        # without forced trailing-newline insertion/removal.
        return f"{instruction} {query}" if instruction else query
    # Canonical keeps wording but applies the visdoc query surface contract:
    # normalize whitespace and preserve a single trailing newline.
    text = f"{instruction} {query}\n" if instruction else f"{query}\n"
    if _visdoc_bool_default(defaults, "canonical_normalize_query_text", True):
        return normalize_text_whitespace(
            text,
            ensure_trailing_newline=_visdoc_bool_default(defaults, "query_append_trailing_newline", True),
        )
    return text


def _compose_candidate_text(*, defaults: dict[str, Any], mode: str) -> str:
    instruction = _visdoc_default(defaults, "candidate_instruction", _VISDOC_DEFAULTS["candidate_instruction"])
    if mode == "origin":
        # Archive parser surface for visdoc candidate is a single-line prompt:
        # "<|image_pad|> instruction".
        return f"{STANDARD_IMAGE_TOKEN} {instruction}" if instruction else STANDARD_IMAGE_TOKEN
    # Canonical keeps the same wording but switches layout to the block form:
    # "<|image_pad|>\ninstruction\n". This captures the documented visdoc
    # visual_token_separator + trailing_newline defaults.
    separator = _visdoc_default(defaults, "candidate_token_separator", "\n")
    suffix = "\n" if _visdoc_bool_default(defaults, "candidate_append_trailing_newline", True) else ""
    if instruction:
        return f"{STANDARD_IMAGE_TOKEN}{separator}{instruction}{suffix}"
    return f"{STANDARD_IMAGE_TOKEN}{suffix}"


def _build_visrag_item(
    sample: dict[str, Any],
    *,
    role: str,
    mode: str,
    defaults: dict[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    if role not in {"query", "candidate"}:
        raise ValueError(f"Unsupported visdoc eval role: {role}")

    if role == "query":
        query = sample.get("query") or sample.get("text") or ""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Missing required raw field for visdoc eval transform: role=query, field=query")
        item: RetrievalQueryItem = {
            "id": str(sample["id"]),
            "query": MultiModalInput(
                text=_compose_query_text(query, defaults=defaults, mode=mode),
                media=[],
            ),
        }
        metadata = extract_item_metadata(
            sample,
            consumed_fields={"id", "query", "text", "images", "image", "media_metadata"},
        )
    else:
        images = [image for image in (sample.get("images") or []) if image is not None]
        if not images and sample.get("image") is not None:
            images = [sample["image"]]
        if not images:
            raise ValueError("Missing required media for visdoc eval transform: role=candidate")
        # Candidate text/media are built together so runtime can enforce
        # visual_token_alignment as an invariant downstream:
        # token count/order/type must stay aligned with media slots.
        item = {
            "id": str(sample["id"]),
            "candidate": MultiModalInput(
                text=_compose_candidate_text(defaults=defaults, mode=mode),
                media=collect_image_media({**sample, "images": images}),
            ),
        }
        metadata = extract_item_metadata(sample, consumed_fields={"id", "images", "image", "text"})

    if metadata:
        item["metadata"] = metadata
    return item


def _build_transform(
    dataset_name: str,
    *,
    role: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _dataset_spec(dataset_name)
    normalized = _normalize_transform_kwargs(dataset_name, transform_kwargs)
    return partial(
        _build_visrag_item,
        role=role,
        mode=str(normalized.get("runtime_mode") or "canonical"),
        defaults=normalized,
    )


def build_query_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    return _build_transform(dataset_name, role="query", transform_kwargs=transform_kwargs)


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    return _build_transform(dataset_name, role="candidate", transform_kwargs=transform_kwargs)


__all__ = [
    "DATASET_NAMES",
    "QUERY_READ_COLUMNS_BY_DATASET",
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "get_query_read_columns",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "build_query_transform",
    "build_candidate_transform",
]
