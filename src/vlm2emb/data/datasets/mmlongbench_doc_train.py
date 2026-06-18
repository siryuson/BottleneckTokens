"""MMLongBench-Doc training dataset runtime."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from vlm2emb.auto import AutoDataset
from vlm2emb.data.conversion.mmlongbench_doc_train import mmlongbench_doc_page_key
from vlm2emb.data.datasets.base import LanceTableExtension, SampleLanceDataset, SampleTransform
from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    canonicalize_multimodal_text,
    decode_image,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    join_instruction_body,
    normalize_instruction_body_separator,
    normalize_instruction_text,
    normalize_side_transform_mapping,
)

SUPPORTED_TASK_ARCHETYPES = ("document_image_retrieval",)
DEFAULT_QUERY_INSTRUCTION = "Find a document image that matches the given query:"
DEFAULT_POSITIVE_INSTRUCTION = "Understand the content of the provided document image."
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}
MMLONGBENCH_DOC_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"instruction", "visual_token_placement", "trailing_newline"},
    "negative": {"trailing_newline", "empty"},
}


@dataclass(frozen=True)
class MMLongBenchDocTrainTransformConfig:
    """Default transform configuration for MMLongBench-Doc training rows."""

    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    query_instruction_body_separator: str = "space"
    query_trailing_newline: str = "ensure_single"
    positive_instruction: str = DEFAULT_POSITIVE_INSTRUCTION
    positive_visual_token_placement: str = "own_line"
    positive_trailing_newline: str = "ensure_single"
    negative_trailing_newline: str = "ensure_single"


def _apply_trailing_newline(text: str, mode: str, *, dataset_name: str) -> str:
    """Apply one side's trailing-newline policy."""

    if mode not in TRAILING_NEWLINE_VALUES:
        raise ValueError(f"{dataset_name} trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}.")
    if mode == "preserve":
        return text
    if mode == "strip":
        return text.rstrip("\n")
    return text.rstrip("\n") + "\n" if text else ""


def _parse_page_index(record: Mapping[str, Any]) -> int:
    """Parse the single evidence page kept by conversion."""

    try:
        pages = ast.literal_eval(str(record.get("evidence_pages") or "[]"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError("MMLongBench-Doc row has invalid evidence_pages.") from exc
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], int):
        raise ValueError("MMLongBench-Doc runtime expects one integer evidence page.")
    return pages[0]


def _page_key_from_record(record: Mapping[str, Any]) -> str:
    """Build the page-table lookup key from a primary row."""

    return mmlongbench_doc_page_key(str(record.get("doc_id") or ""), _parse_page_index(record))


def _normalize_transform_kwargs(values: Mapping[str, Any] | None) -> MMLongBenchDocTrainTransformConfig:
    """Validate YAML-friendly default transform kwargs."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="MMLongBench-Doc",
        allowed_keys=MMLONGBENCH_DOC_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    for side, side_values in (("query", query), ("positive", positive), ("negative", negative)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"MMLongBench-Doc transform.{side}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    placement = positive.get("visual_token_placement")
    if placement is not None and placement != "own_line":
        raise ValueError("MMLongBench-Doc transform.positive.visual_token_placement must be 'own_line'.")
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("MMLongBench-Doc transform.negative.empty must be 'empty_multimodal_input'.")
    return MMLongBenchDocTrainTransformConfig(
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="MMLongBench-Doc",
            name="transform.query.instruction",
            default=DEFAULT_QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="MMLongBench-Doc",
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="MMLongBench-Doc",
            name="transform.positive.instruction",
            default=DEFAULT_POSITIVE_INSTRUCTION,
            allow_empty=False,
        ),
        positive_visual_token_placement="own_line",
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
        negative_trailing_newline=negative.get("trailing_newline", "ensure_single"),
    )


def _positive_text(config: MMLongBenchDocTrainTransformConfig, *, dataset_name: str) -> str:
    """Build the positive document-page text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN}\n{config.positive_instruction}",
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, config.positive_trailing_newline, dataset_name=dataset_name)


def _positive_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the joined PDF page bytes."""

    image = record.get("page_image")
    if not isinstance(image, (bytes, bytearray)):
        raise TypeError("MMLongBench-Doc row is missing required rendered page bytes.")
    return [
        {
            "kind": "image",
            "content": decode_image(bytes(image)),
            "metadata": {
                "path": _page_key_from_record(record),
                "doc_id": str(record.get("doc_id") or ""),
                "page_index": _parse_page_index(record),
            },
        }
    ]


def transform_mmlongbench_doc_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    config: MMLongBenchDocTrainTransformConfig,
) -> dict[str, Any]:
    """Transform one joined MMLongBench-Doc row into a TrainSample."""

    question = normalize_text_whitespace(str(record.get("question") or ""))
    query_text = join_instruction_body(
        config.query_instruction,
        question,
        separator=config.query_instruction_body_separator,
    )
    query_text = _apply_trailing_newline(
        query_text,
        config.query_trailing_newline,
        dataset_name=dataset_name,
    )
    page_index = _parse_page_index(record)
    return {
        "query": {"text": query_text, "media": []},
        "positive": {
            "text": _positive_text(config, dataset_name=dataset_name),
            "media": _positive_media(record),
        },
        "negative": {
            "text": _apply_trailing_newline("", config.negative_trailing_newline, dataset_name=dataset_name),
            "media": [],
        },
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "doc_id": str(record.get("doc_id") or ""),
            "doc_type": str(record.get("doc_type") or ""),
            "page_index": page_index,
            "page_key": _page_key_from_record(record),
            "evidence_pages": str(record.get("evidence_pages") or ""),
            "evidence_sources": str(record.get("evidence_sources") or ""),
            "answer_format": str(record.get("answer_format") or ""),
        },
    }


def build_mmlongbench_doc_train_default_transform(
    *,
    dataset_name: str,
    split: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default MMLongBench-Doc transform."""

    return partial(
        transform_mmlongbench_doc_train_sample,
        dataset_name=dataset_name,
        split=split,
        config=_normalize_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("mmlongbench_doc_train")
class MMLongBenchDocTrainDataset(SampleLanceDataset):
    """MMLongBench-Doc text-to-document-page retrieval training dataset."""

    def __init__(
        self,
        path: str,
        split: str = "official_train_without_mmeb_v2_eval",
        dataset_name: str = "MMLongBench-doc",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("MMLongBenchDocTrainDataset transform must be a callable SampleTransform or None.")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns for MMLongBenchDocTrainDataset.")
        effective_read_columns = read_columns if read_columns is not None else columns
        metadata = dict(metadata or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", split)
        root = Path(path)
        effective_transform = (
            transform
            if transform is not None
            else build_mmlongbench_doc_train_default_transform(
                dataset_name=dataset_name,
                split=split,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(root / "data" / f"{split}.lance"),
            read_columns=effective_read_columns,
            extensions=[
                LanceTableExtension(
                    lance_path=str(root / "data" / "pages.lance"),
                    key=_page_key_from_record,
                    column_name="path",
                    read_columns=["image", "doc_id", "page_index"],
                    column_name_map={"image": "page_image", "doc_id": "page_doc_id"},
                )
            ],
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.dataset_name = dataset_name
        self.split = split
        self.subset = None
        self.variant = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MMLongBenchDocTrainDataset:
        """Build a MMLongBench-Doc dataset from one YAML config entry."""

        return cls(
            path=str(config["path"]),
            split=str(config.get("split", "official_train_without_mmeb_v2_eval")),
            dataset_name=str(config.get("dataset_name", "MMLongBench-doc")),
            read_columns=config.get("read_columns"),
            metadata=config.get("metadata"),
            transform_kwargs=config.get("transform"),
        )


__all__ = [
    "MMLongBenchDocTrainDataset",
    "MMLongBenchDocTrainTransformConfig",
    "build_mmlongbench_doc_train_default_transform",
    "transform_mmlongbench_doc_train_sample",
]
