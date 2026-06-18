"""Temporary MMEB-train expected content helpers for parser tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    normalize_text_whitespace,
)

DEFAULT_SPLIT = "train"


class MmebTrainExpectedSide(TypedDict):
    """One query/positive/negative side in a temporary expected-content table."""

    text: str
    path: str | None
    origin_path: str | None
    has_media: bool


class MmebTrainExpectedRow(TypedDict):
    """One subset row in a temporary expected-content table."""

    subset: str
    source_path: str | None
    source_row: int
    query: MmebTrainExpectedSide
    positive: MmebTrainExpectedSide
    negative: MmebTrainExpectedSide


class MmebTrainExpectedDocument(TypedDict):
    """Top-level temporary expected-content document."""

    split: str
    source_root: str | None
    image_root: str | None
    rows: list[MmebTrainExpectedRow]


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _path_value(value: Any) -> str | None:
    text = _string_value(value).strip()
    return text or None


def _replace_legacy_image_tokens(text: str) -> str:
    normalized = text
    for token in LEGACY_IMAGE_TOKENS:
        normalized = normalized.replace(token, STANDARD_IMAGE_TOKEN)
    return normalized


def _normalize_text_only_query_after_token_removal(text: str) -> str:
    return normalize_text_whitespace(text.strip("\n"), ensure_trailing_newline=True)


def _strip_horizontal_tail(text: str) -> str:
    return text.rstrip(" \t")


def _ensure_single_trailing_newline(text: str) -> str:
    return normalize_text_whitespace(text, ensure_trailing_newline=True)


def _normalize_visual_prompt_block(text: str) -> str:
    if not text.startswith(STANDARD_IMAGE_TOKEN):
        return text
    body = text[len(STANDARD_IMAGE_TOKEN) :].lstrip(" \t\n")
    normalized = f"{STANDARD_IMAGE_TOKEN}\n{body}" if body else STANDARD_IMAGE_TOKEN
    return _ensure_single_trailing_newline(normalized)


def _normalize_visdial_query(text: str) -> str:
    normalized = text.replace("image retrieval: Q:", "image retrieval:\nQ:", 1)
    return _ensure_single_trailing_newline(normalized)


def _origin_full_path(path: str | None, image_root: Path | None) -> str | None:
    if not path or image_root is None:
        return None
    return str(image_root / path)


def _build_side(
    *,
    text: str,
    path: str | None,
    image_root: Path | None,
) -> MmebTrainExpectedSide:
    return {
        "text": text,
        "path": path,
        "origin_path": _origin_full_path(path, image_root),
        "has_media": path is not None,
    }


def build_expected_content_row(
    raw_row: Mapping[str, Any],
    *,
    subset: str,
    source_path: str | Path | None = None,
    source_row: int = 0,
    image_root: str | Path | None = None,
) -> MmebTrainExpectedRow:
    """Build one expected content row for MMEB-train parser tests."""

    query_text = _replace_legacy_image_tokens(_string_value(raw_row.get("qry")))
    positive_text = _replace_legacy_image_tokens(_string_value(raw_row.get("pos_text")))
    negative_text = _replace_legacy_image_tokens(_string_value(raw_row.get("neg_text")))

    query_path = _path_value(raw_row.get("qry_image_path"))
    positive_path = _path_value(raw_row.get("pos_image_path"))
    negative_path = _path_value(raw_row.get("neg_image_path"))

    if (not query_text and not query_path) or (not positive_text and not positive_path):
        raise ValueError(f"MMEB-train row has empty query or positive side: subset={subset}")

    if subset == "WebQA" and query_path is None:
        stripped = query_text.replace(STANDARD_IMAGE_TOKEN, "").replace("<image>", "")
        query_text = _normalize_text_only_query_after_token_removal(stripped)

    if subset == "ChartQA":
        positive_text = _strip_horizontal_tail(positive_text)
        negative_text = _strip_horizontal_tail(negative_text)

    if subset == "VisDial":
        query_text = _normalize_visdial_query(query_text)

    if subset == "VisDial" and negative_path is not None and not negative_text.strip():
        negative_text = STANDARD_IMAGE_TOKEN

    if query_path is not None:
        query_text = _normalize_visual_prompt_block(query_text)
    if positive_path is not None:
        positive_text = _normalize_visual_prompt_block(positive_text)
    if negative_path is not None:
        negative_text = _normalize_visual_prompt_block(negative_text)

    normalized_image_root = Path(image_root) if image_root is not None else None
    return {
        "subset": subset,
        "source_path": str(source_path) if source_path is not None else None,
        "source_row": int(source_row),
        "query": _build_side(text=query_text, path=query_path, image_root=normalized_image_root),
        "positive": _build_side(text=positive_text, path=positive_path, image_root=normalized_image_root),
        "negative": _build_side(text=negative_text, path=negative_path, image_root=normalized_image_root),
    }


def find_subset_parquet(
    source_root: str | Path,
    *,
    subset: str,
    split: str = DEFAULT_SPLIT,
) -> Path:
    subset_dir = Path(source_root) / subset
    candidates = sorted(subset_dir.glob(f"{split}-*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No MMEB-train parquet found for subset={subset}, split={split}")
    return candidates[0]


def discover_subsets(source_root: str | Path, *, split: str = DEFAULT_SPLIT) -> list[str]:
    root = Path(source_root)
    return [
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and any(path.glob(f"{split}-*.parquet"))
    ]


def read_first_parquet_row(path: str | Path) -> dict[str, Any]:
    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index)
        if table.num_rows:
            return dict(table.slice(0, 1).to_pylist()[0])
    raise ValueError(f"Parquet file has no rows: {parquet_path}")


def build_expected_content_document(
    rows: Iterable[MmebTrainExpectedRow],
    *,
    split: str = DEFAULT_SPLIT,
    source_root: str | Path | None = None,
    image_root: str | Path | None = None,
) -> MmebTrainExpectedDocument:
    return {
        "split": split,
        "source_root": str(source_root) if source_root is not None else None,
        "image_root": str(image_root) if image_root is not None else None,
        "rows": list(rows),
    }


def build_expected_content_from_root(
    source_root: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
    subsets: Sequence[str] | None = None,
    image_root: str | Path | None = None,
) -> MmebTrainExpectedDocument:
    root = Path(source_root)
    selected_subsets = list(subsets) if subsets else discover_subsets(root, split=split)
    normalized_image_root = Path(image_root) if image_root is not None else root
    rows: list[MmebTrainExpectedRow] = []
    for subset in selected_subsets:
        parquet_path = find_subset_parquet(root, subset=subset, split=split)
        rows.append(
            build_expected_content_row(
                read_first_parquet_row(parquet_path),
                subset=subset,
                source_path=parquet_path,
                source_row=0,
                image_root=normalized_image_root,
            )
        )
    return build_expected_content_document(
        rows,
        split=split,
        source_root=root,
        image_root=normalized_image_root,
    )


def write_expected_content_document(
    document: MmebTrainExpectedDocument,
    output_path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file_handle:
        json.dump(document, file_handle, ensure_ascii=False, indent=indent)
        file_handle.write("\n")
