"""VisRAG training dataset runtime."""

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
    image_media_from_record,
    join_instruction_body,
    normalize_instruction_body_separator,
    normalize_instruction_text,
    normalize_side_transform_mapping,
)

DATASET_ARCHETYPE = "document_retrieval"
VISRAG_TRANSFORM_KEYS = {
    "query": {"instructions", "instruction_body_separator", "trailing_newline"},
    "positive": {"instructions", "visual_token_placement", "trailing_newline"},
    "negative": {"empty"},
}
INSTRUCTIONS_KEYS = {"by_source", "fallback"}
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}

QUERY_SOURCE_PROMPTS: dict[str, str] = {
    "NeurIPS Papers": (
        "This query is about a research paper from NeurIPS, a leading AI/ML conference. "
        "The document contains technical discussions, methodologies, and findings. "
        "Identify relevant papers and sections that address the query: "
    ),
    "Textbooks": (
        "This query is related to a college-level textbook, which provides structured "
        "explanations, definitions, and examples. Find the most relevant concepts or "
        "explanations that address the query: "
    ),
    "ICML Papers": (
        "This query is about a research paper from ICML, a leading AI/ML conference. "
        "The document contains theoretical insights, experiments, and applications. "
        "Identify relevant papers and sections that best answer the query: "
    ),
    "Manuallib": (
        "This query pertains to a product manual, which contains detailed technical "
        "specifications, usage instructions, and troubleshooting steps. "
        "Find the most relevant section that answers the query: "
    ),
    "ArxivQA": (
        "This query is related to retrieving a relevant figure from an ArXiv research paper. "
        "The retrieved figure should contain scientific plots, mathematical visualizations, "
        "or experimental results that best address the query: "
    ),
    "ChartQA": (
        "This query is related to retrieving a relevant chart that visually represents "
        "numerical or categorical data. The retrieved chart should contain bar graphs, "
        "line charts, or other visual elements necessary to analyze trends, compare values, "
        "or extract insights related to the query: "
    ),
    "MP-DocVQA": (
        "This query is related to retrieving a relevant page from a multi-page document, "
        "such as reports, invoices, or research papers. The retrieved document should "
        "contain text, tables, or structured information necessary to answer the query: "
    ),
    "InfoVQA": (
        "This query is related to retrieving an infographic that visually presents "
        "statistical or factual information using charts, icons, and structured layouts. "
        "The retrieved image should contain the necessary visual elements to provide the "
        "best context for answering the query: "
    ),
    "PlotQA": (
        "This query relates to retrieving a relevant plot or chart that visually represents "
        "numerical data. The retrieved figure should contain the necessary information "
        "to analyze trends, compare values, or extract key insights related to the query: "
    ),
    "SlideVQA": (
        "This query is related to retrieving a relevant presentation slide that visually "
        "presents structured information. The retrieved slide should contain the necessary text, "
        "charts, or graphics to provide the best answer to the query: "
    ),
}

TARGET_SOURCE_PROMPTS: dict[str, str] = {
    "Textbooks": "A textbook page with structured educational content and explanations.",
    "ICML Papers": "A research paper from ICML, covering machine learning topics.",
    "NeurIPS Papers": "A research paper from NeurIPS on AI and ML topics.",
    "Manuallib": "A product manual page with technical specifications and instructions.",
    "InfoVQA": "An infographic with structured data, charts, and annotations.",
    "PlotQA": "A numerical data visualization, such as bar charts or line graphs.",
    "SlideVQA": "A presentation slide with text, bullet points, and diagrams.",
    "ArxivQA": "A figure from a research paper, including plots or experimental results.",
    "MP-DocVQA": "A page from a multi-page document with text or tables.",
    "ChartQA": "A statistical chart comparing values or analyzing trends.",
}


def _normalize_source_instruction_mapping(name: str, value: Any) -> dict[str, str]:
    """Validate one source-specific instruction mapping."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"VisRAG {name} must be a mapping of source -> instruction.")
    normalized: dict[str, str] = {}
    for key, instruction in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"VisRAG {name} contains an invalid source key.")
        if not isinstance(instruction, str):
            raise ValueError(f"VisRAG {name} contains a non-string instruction.")
        normalized[key.strip()] = instruction
    return normalized


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


def _normalize_instructions_config(
    value: Any,
    *,
    side: str,
    defaults: Mapping[str, str],
) -> tuple[dict[str, str], str]:
    """Validate one side's source-specific instruction config."""

    if value is None:
        return dict(defaults), ""
    if not isinstance(value, Mapping):
        raise ValueError(f"VisRAG transform.{side}.instructions must be a mapping.")
    unknown = sorted(set(value) - INSTRUCTIONS_KEYS)
    if unknown:
        raise ValueError(f"Unsupported VisRAG transform.{side}.instructions keys: {unknown}")
    return (
        {
            **defaults,
            **_normalize_source_instruction_mapping(
                f"transform.{side}.instructions.by_source",
                value.get("by_source"),
            ),
        },
        normalize_instruction_text(
            value.get("fallback"),
            dataset_name="VisRAG",
            name=f"transform.{side}.instructions.fallback",
            default="",
        ),
    )


def _normalize_visrag_transform_kwargs(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate VisRAG side-scoped default-transform parameters."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="VisRAG",
        allowed_keys=VISRAG_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    query_instructions, fallback_query_instruction = _normalize_instructions_config(
        query.get("instructions"),
        side="query",
        defaults=QUERY_SOURCE_PROMPTS,
    )
    positive_instructions, fallback_positive_instruction = _normalize_instructions_config(
        positive.get("instructions"),
        side="positive",
        defaults=TARGET_SOURCE_PROMPTS,
    )

    visual_token_placement = positive.get("visual_token_placement")
    if visual_token_placement is not None and visual_token_placement != "own_line":
        raise ValueError("VisRAG visual_token_placement must be 'own_line'.")
    for side_name, side_values in (("query", query), ("positive", positive)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"VisRAG transform.{side_name}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("VisRAG empty must be 'empty_multimodal_input'.")

    return {
        "query_instructions": query_instructions,
        "query_instruction_body_separator": normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="VisRAG",
            name="transform.query.instruction_body_separator",
            default="none",
        ),
        "query_trailing_newline": query.get("trailing_newline", "ensure_single"),
        "positive_instructions": positive_instructions,
        "fallback_query_instruction": fallback_query_instruction,
        "fallback_positive_instruction": fallback_positive_instruction,
        "positive_trailing_newline": positive.get("trailing_newline", "ensure_single"),
    }


def build_visrag_query_text(
    query: Any,
    source: Any,
    *,
    query_instructions: Mapping[str, str] | None = None,
    fallback_query_instruction: str = "",
    instruction_body_separator: str = "none",
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the archive ``visrag`` / ``visrag_single`` text query."""

    query_text = "" if query is None else str(query)
    source_text = "" if source is None else str(source)
    instruction = dict(query_instructions or QUERY_SOURCE_PROMPTS).get(
        source_text,
        fallback_query_instruction,
    )
    text = normalize_text_whitespace(
        join_instruction_body(
            instruction,
            query_text,
            separator=instruction_body_separator,
        )
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="VisRAG")


def build_visrag_positive_text(
    source: Any,
    *,
    positive_instructions: Mapping[str, str] | None = None,
    fallback_positive_instruction: str = "",
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the archive ``visrag`` / ``visrag_single`` document positive text."""

    source_text = "" if source is None else str(source)
    instruction = dict(positive_instructions or TARGET_SOURCE_PROMPTS).get(
        source_text,
        fallback_positive_instruction,
    )
    text = canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN} {instruction}" if instruction else STANDARD_IMAGE_TOKEN,
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="VisRAG")


def build_visrag_image_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build VisRAG document-page image media from a raw Lance row."""

    return image_media_from_record(
        record,
        missing_message="VisRAG row is missing required image bytes",
    )


def transform_visrag_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    query_instructions: Mapping[str, str] | None = None,
    positive_instructions: Mapping[str, str] | None = None,
    fallback_query_instruction: str = "",
    fallback_positive_instruction: str = "",
    query_instruction_body_separator: str = "none",
    query_trailing_newline: str | None = "ensure_single",
    positive_trailing_newline: str | None = "ensure_single",
) -> dict[str, Any]:
    """Transform one raw VisRAG row into a TrainSample."""

    source = record.get("source")
    return {
        "query": {
            "text": build_visrag_query_text(
                record.get("query"),
                source,
                query_instructions=query_instructions,
                fallback_query_instruction=fallback_query_instruction,
                instruction_body_separator=query_instruction_body_separator,
                trailing_newline=query_trailing_newline,
            ),
            "media": [],
        },
        "positive": {
            "text": build_visrag_positive_text(
                source,
                positive_instructions=positive_instructions,
                fallback_positive_instruction=fallback_positive_instruction,
                trailing_newline=positive_trailing_newline,
            ),
            "media": build_visrag_image_media(record),
        },
        "negative": {"text": "", "media": []},
        "metadata": {
            key: value
            for key, value in {
                "dataset_name": dataset_name,
                "source": source,
            }.items()
            if value is not None
        },
    }


def build_visrag_train_default_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the default VisRAG runtime transform."""

    normalized = _normalize_visrag_transform_kwargs(transform_kwargs)
    return partial(
        transform_visrag_train_sample,
        dataset_name=dataset_name,
        **normalized,
    )


@AutoDataset.register("visrag_train")
class VisragTrainDataset(SampleLanceDataset):
    """VisRAG training dataset over raw-preserving Lance rows."""

    def __init__(
        self,
        path: str,
        dataset_name: str = "visrag_train",
        read_columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError("VisragTrainDataset transform must be a callable SampleTransform or None")
        effective_transform = (
            transform
            if transform is not None
            else build_visrag_train_default_transform(
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
    def from_config(cls, config: Mapping[str, Any]) -> VisragTrainDataset:
        """Build a VisRAG dataset from an AutoDataset config mapping."""

        kwargs = dict(config)
        kwargs.pop("type", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("VisRAG config transform must be mapping, callable, or None")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("VisRAG config cannot set both transform and transform_kwargs")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "DATASET_ARCHETYPE",
    "QUERY_SOURCE_PROMPTS",
    "TARGET_SOURCE_PROMPTS",
    "VisragTrainDataset",
    "build_visrag_image_media",
    "build_visrag_positive_text",
    "build_visrag_query_text",
    "build_visrag_train_default_transform",
    "transform_visrag_train_sample",
]
