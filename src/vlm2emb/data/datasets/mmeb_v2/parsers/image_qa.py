from __future__ import annotations

"""MMEB image question-answering parser definitions.

The image-qa family keeps dataset-specific defaults local to this file so the
runtime can treat `transform_kwargs` as an override layer only. `origin` should
match the archive prompt assembly; `canonical` only applies the approved
newline/token normalization flags declared below.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.const import normalize_text_whitespace
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_image_media,
    extract_item_metadata,
    compose_parser_text_with_instruction,
    replace_legacy_image_tokens,
    strip_legacy_image_tokens,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem


def _image_qa_default_transform_kwargs() -> dict[str, Any]:
    """Return dataset-local prompt defaults for archive-aligned image QA.

    These keys remain local parser knobs instead of a global rule engine. They
    encode the currently approved QA query surface:
    - image token on its own line;
    - instruction/body composed by the parser;
    - complete prompt ending with a single trailing newline.
    """
    return {
        "runtime_mode": "canonical",
        "query_instruction_leading_newline": True,
        "query_replace_space_newline": True,
        "query_append_trailing_newline": True,
        "query_append_extra_blank_line": False,
        "canonical_query_whitespace_normalization": False,
    }


DATASET_NAMES: tuple[str, ...] = (
    "VizWiz",
    "ChartQA",
    "ScienceQA",
    "GQA",
    "A-OKVQA",
    "DocVQA",
    "InfographicsVQA",
    "Visual7W",
    "TextVQA",
    "OK-VQA",
)

QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "VizWiz": ("id", "qry_text", "image"),
    "ChartQA": ("id", "qry_text", "image"),
    "ScienceQA": ("id", "qry_text", "image"),
    "GQA": ("id", "qry_text", "image"),
    "A-OKVQA": ("id", "qry_inst", "qry_text", "image"),
    "DocVQA": ("id", "qry_text", "image"),
    "InfographicsVQA": ("id", "qry_text", "image"),
    "Visual7W": ("id", "qry_text", "image"),
    "TextVQA": ("id", "qry_text", "image"),
    "OK-VQA": ("id", "qry_inst", "qry_text", "image"),
}
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    name: ("id", "text") for name in DATASET_NAMES
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    name: _image_qa_default_transform_kwargs() for name in DATASET_NAMES
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET["VizWiz"]["canonical_query_whitespace_normalization"] = True

PATH_REL_BY_DATASET: dict[str, str] = {name: f"image-tasks/{name}" for name in DATASET_NAMES}


def _require_dataset_name(dataset_name: str) -> str:
    if dataset_name not in QUERY_READ_COLUMNS_BY_DATASET:
        raise KeyError(f"Unknown MMEB image_qa dataset: {dataset_name}")
    return dataset_name


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the query-side raw columns for a dataset."""
    return QUERY_READ_COLUMNS_BY_DATASET[_require_dataset_name(dataset_name)]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the candidate-side raw columns for a dataset."""
    return CANDIDATE_READ_COLUMNS_BY_DATASET[_require_dataset_name(dataset_name)]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    """Return a copy of the dataset-local default transform kwargs."""
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[_require_dataset_name(dataset_name)])


def _image_bool_default(defaults: Mapping[str, Any], key: str, fallback: bool) -> bool:
    value = defaults.get(key)
    return fallback if value is None else bool(value)


def _string_scalar(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return _string_scalar(value[0])
    return "" if value is None else str(value)


def _clean_instruction(raw_instruction: str, *, leading_newline: bool = False) -> str:
    # QA instruction fields are semantic text, not token-placement owners.
    # Remove any legacy visual marker before the parser rebuilds the final
    # prompt so token placement stays explicit and parser-owned.
    cleaned = replace_legacy_image_tokens(raw_instruction, replacement="").strip()
    if cleaned and leading_newline:
        return "\n" + cleaned
    return cleaned


def _compose_query_with_instruction(
    *,
    qry_inst: str,
    qry_text: str,
    defaults: Mapping[str, Any],
) -> str:
    # The archive parser already treated the image token as a standalone
    # placeholder followed by a complete prompt block. Keep that ownership in
    # the parser for both `origin` and `canonical`.
    return compose_parser_text_with_instruction(
        qry_text,
        instruction=_clean_instruction(
            qry_inst,
            leading_newline=_image_bool_default(defaults, "query_instruction_leading_newline", True),
        ),
        add_image_token=True,
        replace_space_newline=_image_bool_default(defaults, "query_replace_space_newline", True),
        append_trailing_newline=_image_bool_default(defaults, "query_append_trailing_newline", True),
        append_extra_blank_line=_image_bool_default(defaults, "query_append_extra_blank_line", False),
    )


def _finalize_image_qa_query_text(
    text: str,
    *,
    defaults: Mapping[str, Any],
) -> str:
    runtime_mode = str(defaults.get("runtime_mode") or "canonical")
    normalize_for_canonical = _image_bool_default(
        defaults,
        "canonical_query_whitespace_normalization",
        False,
    )
    if runtime_mode != "canonical" or not normalize_for_canonical:
        return text

    # VizWiz rerun queries occasionally keep stray spaces before the terminal
    # newline, e.g. `"What is the title of this book? \n"`. Canonical mode
    # normalizes that whitespace while preserving the parser-owned newline so
    # the semantic content stays unchanged and the runtime surface matches the
    # approved canonical contract. Remove this only after conversion emits the
    # already-normalized VizWiz query text directly from source.
    return normalize_text_whitespace(
        text,
        ensure_trailing_newline=text.endswith("\n"),
    )


def _transform_image_qa_item(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    role: str,
    transform_kwargs: Mapping[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    if role not in {"query", "candidate"}:
        raise ValueError(f"Unsupported MMEB image_qa role for {dataset_name}: {role}")

    has_media = bool(sample.get("images")) or sample.get("image") is not None
    raw_text = _string_scalar(sample.get("text"))
    qry_text = _string_scalar(sample.get("qry_text"))
    qry_inst = _string_scalar(sample.get("qry_inst"))

    if role == "candidate":
        text = raw_text or _string_scalar(sample.get("tgt_text"))
    elif has_media and qry_inst:
        # Media-backed QA rows keep parser-owned token placement and terminal
        # newline. Canonical-only whitespace cleanup stays a narrow VizWiz
        # exception handled in `_finalize_image_qa_query_text`.
        text = _finalize_image_qa_query_text(
            _compose_query_with_instruction(
                qry_inst=qry_inst,
                qry_text=qry_text,
                defaults=transform_kwargs,
            ),
            defaults=transform_kwargs,
        )
    elif has_media and qry_text:
        # Image-QA rerun artifacts still store legacy image markers, but both
        # the original parser surface and the downstream Qwen stack expect the
        # standardized token. Keep this upgrade enabled in `origin` and
        # `canonical` so parser output follows the historical runtime contract
        # instead of the raw artifact. Remove it only after conversion writes
        # the standardized token surface natively.
        text = _finalize_image_qa_query_text(
            replace_legacy_image_tokens(qry_text),
            defaults=transform_kwargs,
        )
    elif qry_text:
        # Text-only QA fallbacks must not retain a visual placeholder once the
        # media slot disappears. This is an invariant repair, not a style knob.
        text = strip_legacy_image_tokens(qry_text)
    else:
        text = raw_text

    multimodal_input = MultiModalInput(text=text, media=collect_image_media(sample))
    consumed_fields = {
        "id",
        "text",
        "images",
        "image",
        "qry_text",
        "qry_inst",
        "qry_img_path",
        "tgt_text",
        "tgt_inst",
        "tgt_img_path",
        "media_metadata",
    }
    metadata = extract_item_metadata(sample, consumed_fields=consumed_fields)
    if role == "query":
        item: RetrievalQueryItem = {"id": str(sample["id"]), "query": multimodal_input}
    else:
        item = {"id": str(sample["id"]), "candidate": multimodal_input}
    if metadata:
        item["metadata"] = metadata
    return item


def build_query_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    dataset_name = _require_dataset_name(dataset_name)
    normalized_transform_kwargs = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized_transform_kwargs.update(dict(transform_kwargs))
    return partial(
        _transform_image_qa_item,
        dataset_name=dataset_name,
        role="query",
        transform_kwargs=normalized_transform_kwargs,
    )


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    dataset_name = _require_dataset_name(dataset_name)
    normalized_transform_kwargs = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        normalized_transform_kwargs.update(dict(transform_kwargs))
    return partial(
        _transform_image_qa_item,
        dataset_name=dataset_name,
        role="candidate",
        transform_kwargs=normalized_transform_kwargs,
    )


__all__ = [
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DATASET_NAMES",
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "PATH_REL_BY_DATASET",
    "QUERY_READ_COLUMNS_BY_DATASET",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "get_query_read_columns",
    "build_candidate_transform",
    "build_query_transform",
]
