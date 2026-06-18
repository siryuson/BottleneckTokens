from __future__ import annotations

"""MMEB image-to-image / visual-grounding parser definitions.

This family still uses local transform kwargs instead of a shared rule engine.
That is intentional: the parser owns the final runtime surface and keeps the
archive-aligned `origin` behavior next to the approved `plan canonical`
defaults. The older key names remain local implementation knobs only; they are
not a project-wide rule vocabulary.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from vlm2emb.data.datasets.base import SampleTransform
from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN
from vlm2emb.data.datasets.mmeb_v2.parsers.shared import (
    collect_image_media,
    extract_item_metadata,
    compose_parser_text_with_instruction,
    replace_legacy_image_tokens,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem

PARSER_KEY = "image_i2i_vg"


DATASET_NAMES: tuple[str, ...] = (
    "NIGHTS",
    "OVEN",
    "FashionIQ",
    "CIRR",
    "Visual7W-Pointing",
    "MSCOCO",
    "RefCOCO",
    "RefCOCO-Matching",
)

PATH_REL_BY_DATASET: dict[str, str] = {
    name: f"image-tasks/{name}"
    for name in DATASET_NAMES
}
MODALITY_BY_DATASET: dict[str, str] = {name: "image" for name in DATASET_NAMES}
TASK_TYPE_BY_DATASET: dict[str, str] = {
    "NIGHTS": "image_retrieval",
    "OVEN": "image_retrieval",
    "FashionIQ": "image_retrieval",
    "CIRR": "image_retrieval",
    "Visual7W-Pointing": "image_visual_grounding",
    "MSCOCO": "image_visual_grounding",
    "RefCOCO": "image_visual_grounding",
    "RefCOCO-Matching": "image_visual_grounding",
}

QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "NIGHTS": ("id", "qry_text", "image"),
    "OVEN": ("id", "qry_text", "image"),
    "FashionIQ": ("id", "qry_text", "image"),
    "CIRR": ("id", "qry_inst", "qry_text", "image"),
    "Visual7W-Pointing": ("id", "qry_text", "image"),
    "MSCOCO": ("id", "qry_text", "image"),
    "RefCOCO": ("id", "qry_text", "image"),
    "RefCOCO-Matching": ("id", "qry_text", "image"),
}
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    name: ("id", "tgt_text", "image") for name in DATASET_NAMES
}
DEFAULT_TRANSFORM_KWARGS: dict[str, Any] = {
    "runtime_mode": "canonical",
    # These parser-local knobs map to the project-level surface rules without
    # forcing every parser to expose the full global vocabulary:
    # - `query_*` controls image-token placement plus the prompt tail;
    # - `target_*` controls candidate block layout only when the parser owns
    #   that candidate prompt template.
    "query_instruction_leading_newline": True,
    "query_replace_space_newline": True,
    "query_append_trailing_newline": True,
    "query_append_extra_blank_line": False,
    "target_include_caption_when_present": True,
    "target_replace_space_newline": True,
    "target_append_trailing_newline_with_caption": True,
    "target_append_trailing_newline_without_caption": False,
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    name: dict(DEFAULT_TRANSFORM_KWARGS)
    for name in DATASET_NAMES
}
for _dataset_name in ("Visual7W-Pointing", "MSCOCO", "RefCOCO"):
    DEFAULT_TRANSFORM_KWARGS_BY_DATASET[_dataset_name][
        "canonical_cropped_object_candidate_sentence"
    ] = True
del _dataset_name

CROPPED_OBJECT_CANDIDATE_BODY = "Represent the given cropped image of the object"


def _require_dataset_name(dataset_name: str) -> None:
    if dataset_name not in DATASET_NAMES:
        raise KeyError(f"Unknown MMEB image_i2i_vg dataset: {dataset_name}")


def get_query_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the query-side raw columns for a dataset."""
    _require_dataset_name(dataset_name)
    return QUERY_READ_COLUMNS_BY_DATASET[dataset_name]


def get_candidate_read_columns(dataset_name: str) -> tuple[str, ...]:
    """Return the candidate-side raw columns for a dataset."""
    _require_dataset_name(dataset_name)
    return CANDIDATE_READ_COLUMNS_BY_DATASET[dataset_name]


def get_default_transform_kwargs(dataset_name: str) -> dict[str, Any]:
    """Return a copy of the dataset-local default transform kwargs."""
    _require_dataset_name(dataset_name)
    return dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])


def _merge_transform_kwargs(
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(DEFAULT_TRANSFORM_KWARGS_BY_DATASET[dataset_name])
    if transform_kwargs:
        merged.update(dict(transform_kwargs))
    return merged


def _bool_default(transform_kwargs: dict[str, Any], key: str, fallback: bool) -> bool:
    value = transform_kwargs.get(key)
    return fallback if value is None else bool(value)


def _is_canonical(transform_kwargs: Mapping[str, Any]) -> bool:
    return str(transform_kwargs.get("runtime_mode", "canonical")) == "canonical"


def _clean_instruction(raw_instruction: str, *, leading_newline: bool = False) -> str:
    # Instruction fields do not own visual token placement. Normalize any
    # legacy marker out of the instruction before rebuilding the parser-owned
    # prompt surface.
    cleaned = replace_legacy_image_tokens(raw_instruction, replacement="").strip()
    if cleaned and leading_newline:
        return "\n" + cleaned
    return cleaned


def _compose_query_with_instruction(
    *,
    qry_inst: str,
    qry_text: str,
    transform_kwargs: dict[str, Any],
) -> str:
    # Image-I2I / grounding queries own their final prompt surface here. The
    # parser keeps archive-aligned `origin` defaults and parser-owned canonical
    # defaults without relying on loader-time rewrites.
    return compose_parser_text_with_instruction(
        qry_text,
        instruction=_clean_instruction(
            qry_inst,
            leading_newline=_bool_default(transform_kwargs, "query_instruction_leading_newline", False),
        ),
        add_image_token=True,
        replace_space_newline=_bool_default(transform_kwargs, "query_replace_space_newline", True),
        append_trailing_newline=_bool_default(transform_kwargs, "query_append_trailing_newline", True),
        append_extra_blank_line=_bool_default(transform_kwargs, "query_append_extra_blank_line", False),
    )


def _compose_candidate_with_instruction(
    *,
    tgt_inst: str,
    tgt_text: str,
    transform_kwargs: dict[str, Any],
) -> str:
    include_caption = _bool_default(transform_kwargs, "target_include_caption_when_present", True) and bool(tgt_text.strip())
    body = tgt_text if include_caption else ""
    append_trailing_newline = _bool_default(
        transform_kwargs,
        "target_append_trailing_newline_with_caption" if include_caption else "target_append_trailing_newline_without_caption",
        include_caption,
    )
    return compose_parser_text_with_instruction(
        body,
        instruction=_clean_instruction(
            tgt_inst,
            leading_newline=_bool_default(transform_kwargs, "target_instruction_leading_newline", False),
        ),
        add_image_token=True,
        replace_space_newline=_bool_default(transform_kwargs, "target_replace_space_newline", True),
        append_trailing_newline=append_trailing_newline,
        append_extra_blank_line=_bool_default(transform_kwargs, "target_append_extra_blank_line", False),
    )


def _canonicalize_cropped_object_candidate_sentence(text: str) -> tuple[str, bool]:
    """Make the cropped-object image candidate a complete prompt block."""

    normalized = replace_legacy_image_tokens(text).rstrip("\n")
    expected_prefix = f"{STANDARD_IMAGE_TOKEN}\n"
    if not normalized.startswith(expected_prefix):
        return text, False
    body = normalized[len(expected_prefix) :].strip()
    if body not in {CROPPED_OBJECT_CANDIDATE_BODY, f"{CROPPED_OBJECT_CANDIDATE_BODY}."}:
        return text, False
    canonical = f"{expected_prefix}{CROPPED_OBJECT_CANDIDATE_BODY}.\n"
    return canonical, canonical != text


def _compose_query_text(sample: dict[str, Any], *, transform_kwargs: dict[str, Any]) -> str:
    qry_text = str(sample.get("qry_text") or "")
    qry_inst = str(sample.get("qry_inst") or "")
    has_media = bool(sample.get("images")) or sample.get("image") is not None
    if has_media and qry_inst:
        return _compose_query_with_instruction(
            qry_inst=qry_inst,
            qry_text=qry_text,
            transform_kwargs=transform_kwargs,
        )
    if has_media and qry_text:
        return replace_legacy_image_tokens(qry_text)
    return qry_text or str(sample.get("text") or "")


def _compose_candidate_text(
    sample: dict[str, Any],
    *,
    transform_kwargs: dict[str, Any],
) -> tuple[str, list[str]]:
    surface_repairs: list[str] = []
    tgt_text = str(sample.get("tgt_text") or "")
    tgt_inst = str(sample.get("tgt_inst") or "")
    has_media = bool(sample.get("images")) or sample.get("image") is not None
    if has_media and tgt_inst:
        # Candidate prompts only switch to parser-owned block layout when the
        # subset actually provides an instruction/body template. Plain cropped
        # object labels and grounding references remain direct text passthrough.
        text = _compose_candidate_with_instruction(
            tgt_inst=tgt_inst,
            tgt_text=tgt_text,
            transform_kwargs=transform_kwargs,
        )
    elif has_media and tgt_text:
        text = replace_legacy_image_tokens(tgt_text)
    else:
        text = tgt_text or str(sample.get("text") or "")

    if _is_canonical(transform_kwargs) and _bool_default(
        transform_kwargs,
        "canonical_cropped_object_candidate_sentence",
        False,
    ):
        canonical, changed = _canonicalize_cropped_object_candidate_sentence(text)
        if changed:
            text = canonical
            surface_repairs.append("canonical_cropped_object_candidate_sentence")
    return text, surface_repairs


def _add_surface_repairs(metadata: dict[str, Any], surface_repairs: list[str]) -> None:
    if surface_repairs:
        metadata["surface_repairs"] = surface_repairs


def _transform_query_item(
    sample: dict[str, Any],
    *,
    subset_name: str,
    transform_kwargs: dict[str, Any],
) -> RetrievalQueryItem:
    del subset_name
    multimodal_input = MultiModalInput(
        text=_compose_query_text(sample, transform_kwargs=transform_kwargs),
        media=collect_image_media(sample),
    )
    consumed_fields = {
        "id", "text", "images", "image", "qry_text", "qry_inst", "qry_img_path",
        "tgt_text", "tgt_inst", "tgt_img_path", "media_metadata",
    }
    item: RetrievalQueryItem = {"id": str(sample["id"]), "query": multimodal_input}
    metadata = extract_item_metadata(sample, consumed_fields=consumed_fields)
    if metadata:
        item["metadata"] = metadata
    return item


def _transform_candidate_item(
    sample: dict[str, Any],
    *,
    subset_name: str,
    transform_kwargs: dict[str, Any],
) -> RetrievalCandidateItem:
    del subset_name
    text, surface_repairs = _compose_candidate_text(sample, transform_kwargs=transform_kwargs)
    multimodal_input = MultiModalInput(
        text=text,
        media=collect_image_media(sample),
    )
    consumed_fields = {
        "id", "text", "images", "image", "qry_text", "qry_inst", "qry_img_path",
        "tgt_text", "tgt_inst", "tgt_img_path", "media_metadata",
    }
    item: RetrievalCandidateItem = {"id": str(sample["id"]), "candidate": multimodal_input}
    metadata = extract_item_metadata(sample, consumed_fields=consumed_fields)
    _add_surface_repairs(metadata, surface_repairs)
    if metadata:
        item["metadata"] = metadata
    return item


def build_query_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    merged_transform_kwargs = _merge_transform_kwargs(dataset_name, transform_kwargs)
    return partial(
        _transform_query_item,
        subset_name=dataset_name,
        transform_kwargs=merged_transform_kwargs,
    )


def build_candidate_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    _require_dataset_name(dataset_name)
    merged_transform_kwargs = _merge_transform_kwargs(dataset_name, transform_kwargs)
    return partial(
        _transform_candidate_item,
        subset_name=dataset_name,
        transform_kwargs=merged_transform_kwargs,
    )


__all__ = [
    "CANDIDATE_READ_COLUMNS_BY_DATASET",
    "DEFAULT_TRANSFORM_KWARGS_BY_DATASET",
    "DEFAULT_TRANSFORM_KWARGS",
    "MODALITY_BY_DATASET",
    "PATH_REL_BY_DATASET",
    "TASK_TYPE_BY_DATASET",
    "QUERY_READ_COLUMNS_BY_DATASET",
    "get_candidate_read_columns",
    "get_default_transform_kwargs",
    "get_query_read_columns",
    "build_candidate_transform",
    "build_query_transform",
]
