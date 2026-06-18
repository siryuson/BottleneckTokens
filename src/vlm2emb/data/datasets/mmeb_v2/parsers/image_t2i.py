from __future__ import annotations

"""MMEB image-to-image retrieval / dialog-to-image parser definitions.

This family carries approved subset-scoped canonical rewrites:
- `VisDial` changes the instruction/body separator for dialog-style prompts.
- `VisDial` also enforces a canonical candidate block layout.

`WebQA` / `EDIS` are contract-driven instead of style-driven: text-only query
rows without media must strip legacy visual tokens in both `origin` and
`canonical`, because runtime retrieval items cannot claim visual input when no
media slot exists.
All other datasets keep the archive prompt shape and only normalize legacy
tokens to the standard runtime token.
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
    strip_legacy_image_tokens,
)
from vlm2emb.data.schema import MultiModalInput, RetrievalCandidateItem, RetrievalQueryItem


def _image_t2i_default_transform_kwargs() -> dict[str, Any]:
    """Return dataset-local prompt defaults for image-backed retrieval.

    These kwargs stay local to the parser. They encode the approved retrieval
    surface without introducing a shared rule engine:
    - query prompt blocks may own a terminal newline;
    - image-backed target prompts may use parser-owned block layout;
    - text-only rows still defer to explicit alignment repairs below.
    """
    return {
        "runtime_mode": "canonical",
        "query_append_trailing_newline": True,
        "target_include_caption_when_present": True,
        "target_replace_space_newline": True,
        "target_append_trailing_newline_with_caption": True,
        "target_append_trailing_newline_without_caption": True,
    }


def _visdial_default_transform_kwargs() -> dict[str, Any]:
    """Return VisDial-specific canonical overrides on top of the base set.

    VisDial is an approved exception where canonical changes the
    instruction/body separator and trims the extra blank tail inherited from
    the archive dialog surface.
    """
    defaults = _image_t2i_default_transform_kwargs()
    defaults.update(
        {
            "canonical_trim_single_extra_blank_line": True,
            "canonical_dialog_instruction_newline": True,
            "canonical_candidate_block_layout": True,
        }
    )
    return defaults


def _text_only_query_alignment_defaults() -> dict[str, Any]:
    """Return alignment defaults for text-only query subsets.

    WebQA and EDIS do not treat this as a canonical style option. They always
    repair query text so runtime visual-token/media alignment stays valid when
    the row has no media slot.
    """
    defaults = _image_t2i_default_transform_kwargs()
    defaults.update({"enforce_text_only_query_visual_token_alignment": True})
    return defaults


DATASET_NAMES: tuple[str, ...] = (
    "VisualNews_t2i",
    "Wiki-SS-NQ",
    "EDIS",
    "WebQA",
    "MSCOCO_t2i",
    "VisDial",
)

QUERY_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "VisualNews_t2i": ("id", "qry_text"),
    "Wiki-SS-NQ": ("id", "qry_text"),
    "EDIS": ("id", "qry_text"),
    "WebQA": ("id", "qry_text"),
    "MSCOCO_t2i": ("id", "qry_text"),
    "VisDial": ("id", "qry_inst", "qry_text"),
}
CANDIDATE_READ_COLUMNS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "VisualNews_t2i": ("id", "tgt_text", "image"),
    "Wiki-SS-NQ": ("id", "tgt_text", "image"),
    "EDIS": ("id", "tgt_text", "image"),
    "WebQA": ("id", "tgt_inst", "tgt_text", "image"),
    "MSCOCO_t2i": ("id", "tgt_text", "image"),
    "VisDial": ("id", "tgt_inst", "tgt_text", "tgt_img_path", "image"),
}
DEFAULT_TRANSFORM_KWARGS_BY_DATASET: dict[str, dict[str, Any]] = {
    "VisualNews_t2i": _image_t2i_default_transform_kwargs(),
    "Wiki-SS-NQ": _image_t2i_default_transform_kwargs(),
    "EDIS": _text_only_query_alignment_defaults(),
    "WebQA": _text_only_query_alignment_defaults(),
    "MSCOCO_t2i": _image_t2i_default_transform_kwargs(),
    "VisDial": _visdial_default_transform_kwargs(),
}

PATH_REL_BY_DATASET: dict[str, str] = {name: f"image-tasks/{name}" for name in DATASET_NAMES}


def _require_dataset_name(dataset_name: str) -> str:
    if dataset_name not in QUERY_READ_COLUMNS_BY_DATASET:
        raise KeyError(f"Unknown MMEB image_t2i dataset: {dataset_name}")
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


def _canonicalize_dialog_instruction_separator(text: str) -> str:
    """Move dialog prompts to a newline-separated instruction/body boundary."""
    prefix, separator, body = text.partition(": ")
    if not separator:
        return text
    if body.startswith("Q:") or "\nQ:" in body or "\nA:" in body:
        return f"{prefix}:\n{body}"
    return text


def _clean_instruction(raw_instruction: str, *, leading_newline: bool = False) -> str:
    cleaned = replace_legacy_image_tokens(raw_instruction, replacement="").strip()
    if cleaned and leading_newline:
        return "\n" + cleaned
    return cleaned


def _canonicalize_image_token_block_layout(text: str) -> str:
    """Canonicalize image-token prompts to `<image_token>\\n<body>` block layout."""
    normalized = replace_legacy_image_tokens(text)
    if not normalized.startswith(STANDARD_IMAGE_TOKEN):
        return normalized
    body = normalized[len(STANDARD_IMAGE_TOKEN) :].lstrip(" \n")
    had_trailing_newline = normalized.endswith("\n")
    if not body:
        return f"{STANDARD_IMAGE_TOKEN}\n"
    block = f"{STANDARD_IMAGE_TOKEN}\n{body.rstrip(chr(10))}"
    if had_trailing_newline:
        return block + "\n"
    return block


def _compose_target_with_instruction(
    *,
    tgt_inst: str,
    tgt_text: str,
    defaults: Mapping[str, Any],
) -> str:
    include_caption = _image_bool_default(defaults, "target_include_caption_when_present", True) and bool(
        tgt_text.strip()
    )
    body = tgt_text if include_caption else ""
    append_trailing_newline = _image_bool_default(
        defaults,
        "target_append_trailing_newline_with_caption"
        if include_caption
        else "target_append_trailing_newline_without_caption",
        include_caption,
    )
    return compose_parser_text_with_instruction(
        body,
        instruction=_clean_instruction(
            tgt_inst,
            leading_newline=_image_bool_default(defaults, "target_instruction_leading_newline", False),
        ),
        add_image_token=True,
        replace_space_newline=_image_bool_default(defaults, "target_replace_space_newline", True),
        append_trailing_newline=append_trailing_newline,
        append_extra_blank_line=_image_bool_default(defaults, "target_append_extra_blank_line", False),
    )


def _transform_image_t2i_item(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    role: str,
    transform_kwargs: Mapping[str, Any],
) -> RetrievalQueryItem | RetrievalCandidateItem:
    validation_overrides: dict[str, Any] = {}
    surface_repairs: list[str] = []
    if role not in {"query", "candidate"}:
        raise ValueError(f"Unsupported MMEB image_t2i role for {dataset_name}: {role}")

    is_canonical = str(transform_kwargs.get("runtime_mode", "canonical")) == "canonical"
    has_media = bool(sample.get("images")) or sample.get("image") is not None
    raw_text = _string_scalar(sample.get("text"))
    qry_text = _string_scalar(sample.get("qry_text"))
    tgt_text = _string_scalar(sample.get("tgt_text"))
    tgt_inst = _string_scalar(sample.get("tgt_inst"))

    # Image T2I artifacts are not fully uniform across subsets: some candidates
    # carry `tgt_inst + tgt_text`, some only keep `tgt_text`, and some legacy
    # rows fall back to query/raw text fields. Keep the fallback chain explicit
    # here so dataset-specific prompt ownership stays in the parser instead of
    # being hidden behind a generic text resolver.
    if role == "candidate":
        if has_media and tgt_inst:
            text = _compose_target_with_instruction(
                tgt_inst=tgt_inst,
                tgt_text=tgt_text,
                defaults=transform_kwargs,
            )
        elif has_media and tgt_text:
            text = replace_legacy_image_tokens(tgt_text)
        elif qry_text:
            text = qry_text
        elif tgt_text:
            text = tgt_text
        else:
            text = raw_text
        if is_canonical and _image_bool_default(transform_kwargs, "canonical_candidate_block_layout", False):
            # VisDial is the only approved image-T2I subset where canonical
            # upgrades the archive inline token surface to a parser-owned block
            # layout. Keep that rewrite explicit and provenance-tracked.
            block_text = _canonicalize_image_token_block_layout(text)
            if block_text != text:
                surface_repairs.append("canonical_candidate_block_layout")
                text = block_text
    else:
        query_text = qry_text or raw_text
        # Only the approved dialog family (VisDial) opts into this separator fix.
        if is_canonical and _image_bool_default(transform_kwargs, "canonical_trim_single_extra_blank_line", False) and query_text.endswith(
            "\n\n"
        ):
            query_text = query_text[:-1]
        if is_canonical and _image_bool_default(transform_kwargs, "canonical_dialog_instruction_newline", False):
            query_text = _canonicalize_dialog_instruction_separator(query_text)
        # Query-side token handling is dataset-sensitive: archive origin keeps
        # the query token surface for media-backed rows. For text-only rows
        # without media, WebQA/EDIS must strip legacy visual tokens in both
        # modes to satisfy runtime visual-token/media alignment contract.
        if has_media:
            text = replace_legacy_image_tokens(query_text)
        else:
            if _image_bool_default(
                transform_kwargs,
                "enforce_text_only_query_visual_token_alignment",
                False,
            ):
                text = strip_legacy_image_tokens(query_text)
                if text != query_text:
                    surface_repairs.append("visual_token_alignment_strip_without_media")
            else:
                text = replace_legacy_image_tokens(query_text)
                if text != query_text:
                    validation_overrides["allow_visual_token_without_media"] = True

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
    if surface_repairs:
        # Provenance is explicit: keep a machine-readable trace whenever runtime
        # applies non-origin rewrites or invariant repairs.
        metadata["surface_repairs"] = surface_repairs
    if role == "query":
        item: RetrievalQueryItem = {"id": str(sample["id"]), "query": multimodal_input}
    else:
        item = {"id": str(sample["id"]), "candidate": multimodal_input}
    if metadata:
        item["metadata"] = metadata
    if validation_overrides:
        item["_validation"] = validation_overrides
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
        _transform_image_t2i_item,
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
        _transform_image_t2i_item,
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
