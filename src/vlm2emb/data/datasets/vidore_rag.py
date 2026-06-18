"""ViDoRe source-specific RAG-view training dataset."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import SampleLanceDataset, SampleTransform
from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    canonicalize_multimodal_text,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    join_instruction_body,
    normalize_instruction_body_separator,
    normalize_instruction_text,
    normalize_side_transform_mapping,
)
from vlm2emb.data.datasets.vidore import build_vidore_image_media

DATASET_ARCHETYPE = "document_retrieval"
QUERY_INSTRUCTION = "Find a document image that matches the given query:"
TARGET_INSTRUCTION = "Understand the content of the provided document image."
VIDORE_RAG_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"instruction", "visual_token_placement", "trailing_newline"},
    "negative": {"empty"},
}
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}


def _apply_trailing_newline(text: str, mode: Any, *, dataset_name: str) -> str:
    """Apply one side's trailing-newline policy after text normalization."""

    if mode is None or mode == "preserve":
        return text
    if mode == "strip":
        return text.rstrip("\n")
    if mode == "ensure_single":
        return text.rstrip("\n") + "\n" if text else ""
    raise ValueError(
        f"{dataset_name} trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
    )


def _normalize_vidore_rag_transform_kwargs(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate ViDoRe RAG side-scoped default-transform parameters."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="ViDoRe RAG",
        allowed_keys=VIDORE_RAG_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})

    visual_token_placement = positive.get("visual_token_placement")
    if visual_token_placement is not None and visual_token_placement != "own_line":
        raise ValueError("ViDoRe RAG visual_token_placement must be 'own_line'.")
    for side_name, side_values in (("query", query), ("positive", positive)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"ViDoRe RAG transform.{side_name}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("ViDoRe RAG empty must be 'empty_multimodal_input'.")

    return {
        "query_instruction": normalize_instruction_text(
            query.get("instruction"),
            dataset_name="ViDoRe RAG",
            name="transform.query.instruction",
            default=QUERY_INSTRUCTION,
            allow_empty=False,
        ),
        "query_instruction_body_separator": normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="ViDoRe RAG",
            name="transform.query.instruction_body_separator",
            default="space",
        ),
        "query_trailing_newline": query.get("trailing_newline", "ensure_single"),
        "positive_instruction": normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="ViDoRe RAG",
            name="transform.positive.instruction",
            default=TARGET_INSTRUCTION,
            allow_empty=False,
        ),
        "positive_trailing_newline": positive.get("trailing_newline", "ensure_single"),
    }


def build_vidore_rag_query_text(
    query: Any,
    *,
    query_instruction: str = QUERY_INSTRUCTION,
    instruction_body_separator: str = "space",
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the archive ``vidore_rag_single`` text-only query."""

    query_text = "" if query is None else str(query)
    text = normalize_text_whitespace(
        join_instruction_body(
            query_instruction,
            query_text,
            separator=instruction_body_separator,
        )
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="ViDoRe RAG")


def build_vidore_rag_positive_text(
    *,
    positive_instruction: str = TARGET_INSTRUCTION,
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the archive ``vidore_rag_single`` document-image positive text."""

    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN} {positive_instruction}",
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="ViDoRe RAG")


def transform_vidore_rag_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    query_instruction: str = QUERY_INSTRUCTION,
    query_instruction_body_separator: str = "space",
    query_trailing_newline: str | None = "ensure_single",
    positive_instruction: str = TARGET_INSTRUCTION,
    positive_trailing_newline: str | None = "ensure_single",
) -> dict[str, Any]:
    """Transform one raw ViDoRe row into a source-specific RAG TrainSample."""

    return {
        "query": {
            "text": build_vidore_rag_query_text(
                record.get("query"),
                query_instruction=query_instruction,
                instruction_body_separator=query_instruction_body_separator,
                trailing_newline=query_trailing_newline,
            ),
            "media": [],
        },
        "positive": {
            "text": build_vidore_rag_positive_text(
                positive_instruction=positive_instruction,
                trailing_newline=positive_trailing_newline,
            ),
            "media": build_vidore_image_media(record),
        },
        "negative": {"text": "", "media": []},
        "metadata": {
            key: value
            for key, value in {
                "dataset_name": dataset_name,
                "source": record.get("source"),
                "page": record.get("page"),
                "model": record.get("model"),
                "answer_type": record.get("answer_type"),
                "image_filename": record.get("image_filename"),
            }.items()
            if value is not None
        },
    }


def build_vidore_rag_train_default_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the default ViDoRe RAG-view runtime transform."""

    normalized = _normalize_vidore_rag_transform_kwargs(transform_kwargs)
    return partial(
        transform_vidore_rag_train_sample,
        dataset_name=dataset_name,
        **normalized,
    )


@AutoDataset.register("vidore_rag_train")
class VidoreRagTrainDataset(SampleLanceDataset):
    """ViDoRe source-specific RAG-view training dataset."""

    def __init__(
        self,
        path: str,
        dataset_name: str = "vidore_rag_train",
        read_columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("VidoreRagTrainDataset transform must be a callable SampleTransform or None")
        effective_transform = (
            transform
            if transform is not None
            else build_vidore_rag_train_default_transform(
                dataset_name=dataset_name,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(Path(path) / "data" / "train.lance"),
            read_columns=read_columns,
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.dataset_name = dataset_name
        self.transform_kwargs = dict(transform_kwargs or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VidoreRagTrainDataset:
        """Build a ViDoRe RAG-view dataset from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("ViDoRe RAG config transform must be mapping, callable, or None")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("ViDoRe RAG config cannot set both transform and transform_kwargs")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DATASET_ARCHETYPE",
    "QUERY_INSTRUCTION",
    "TARGET_INSTRUCTION",
    "VidoreRagTrainDataset",
    "build_vidore_rag_positive_text",
    "build_vidore_rag_query_text",
    "build_vidore_rag_train_default_transform",
    "transform_vidore_rag_train_sample",
]
