"""ViDoRe training dataset runtime."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import (
    ResolvedTrainDatasetSpec,
    SampleLanceDataset,
    SampleTransform,
)
from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    canonicalize_multimodal_text,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    image_media_from_record,
    normalize_side_transform_mapping,
)

DATASET_ARCHETYPE = "document_retrieval"
VIDORE_TRANSFORM_KEYS = {
    "query": {"visual_token_placement", "trailing_newline"},
    "positive": {"trailing_newline"},
    "negative": {"empty"},
}
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}


def _parse_options(options: Any) -> list[str] | None:
    """Parse ViDoRe options stored either as a list or a stringified list."""

    if options is None or options == "":
        return None
    if isinstance(options, list):
        return [str(option) for option in options]
    if isinstance(options, str):
        try:
            parsed = ast.literal_eval(options)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, list):
            return [str(option) for option in parsed]
    return None


def transform_vidore_answer(answer: Any, options: Any) -> str:
    """Apply the archive ``vidore_fixoptions_v2`` answer normalization."""

    text = "" if answer is None else str(answer)
    parsed_options = _parse_options(options)
    if not parsed_options:
        return normalize_text_whitespace(text)
    old_answer = text
    index_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    if text in index_map and index_map[text] < len(parsed_options):
        text = parsed_options[index_map[text]]
    text = text.replace(f"{old_answer}. ", "").replace(f"{old_answer}) ", "")
    return normalize_text_whitespace(text)


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


def _normalize_vidore_transform_kwargs(values: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Validate ViDoRe side-scoped default-transform parameters."""

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="ViDoRe",
        allowed_keys=VIDORE_TRANSFORM_KEYS,
    )
    for _side, side_values in normalized.items():
        visual_token_placement = side_values.get("visual_token_placement")
        if visual_token_placement is not None and visual_token_placement != "own_line":
            raise ValueError("ViDoRe visual_token_placement must be 'own_line'.")
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"ViDoRe trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
            )
        empty = side_values.get("empty")
        if empty is not None and empty != "empty_multimodal_input":
            raise ValueError("ViDoRe empty must be 'empty_multimodal_input'.")
    return normalized


def _apply_vidore_side_rules(text: str, rules: Mapping[str, Any]) -> str:
    """Apply side-scoped runtime-surface rules without changing parser semantics."""

    return _apply_trailing_newline(
        text,
        rules.get("trailing_newline"),
        dataset_name="ViDoRe",
    )


def build_vidore_full_query_text(query: Any, options: Any) -> str:
    """Build the archive ``vidore_fixoptions_v2`` full-view query text."""

    text = "" if query is None else str(query)
    parsed_options = _parse_options(options)
    if parsed_options:
        options_text = options if isinstance(options, str) else parsed_options
        text = f"{text} Options: {options_text}"
    text = normalize_text_whitespace(text)
    return canonicalize_multimodal_text(
        f"{STANDARD_IMAGE_TOKEN} {text}" if text else STANDARD_IMAGE_TOKEN,
        token=STANDARD_IMAGE_TOKEN,
        legacy_tokens=LEGACY_IMAGE_TOKENS,
    )


def build_vidore_image_media(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build ViDoRe document-page image media from a raw Lance row."""

    return image_media_from_record(
        record,
        fallback_path_fields=("image_filename",),
        missing_message="ViDoRe row is missing required image bytes",
    )


def transform_vidore_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    transform_rules: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Transform one raw ViDoRe full-view row into a TrainSample."""

    rules = transform_rules or {}
    return {
        "query": {
            "text": _apply_vidore_side_rules(
                build_vidore_full_query_text(record.get("query"), record.get("options")),
                rules.get("query", {}),
            ),
            "media": build_vidore_image_media(record),
        },
        "positive": {
            "text": _apply_vidore_side_rules(
                transform_vidore_answer(record.get("answer"), record.get("options")),
                rules.get("positive", {}),
            ),
            "media": [],
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


def build_vidore_train_default_transform(
    *,
    dataset_name: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the default ViDoRe full-view runtime transform."""

    return partial(
        transform_vidore_train_sample,
        dataset_name=dataset_name,
        transform_rules=_normalize_vidore_transform_kwargs(transform_kwargs),
    )


@AutoDataset.register("vidore_train")
class VidoreTrainDataset(SampleLanceDataset):
    """ViDoRe full-view training dataset over raw-preserving Lance rows."""

    def __init__(
        self,
        path: str | None = None,
        dataset_name: str = "vidore_train",
        lance_path: str | None = None,
        read_columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if lance_path is None:
            if path is None:
                raise ValueError("Either path or lance_path must be provided for VidoreTrainDataset.")
            lance_path = str(Path(path) / "data" / "train.lance")
        if transform is not None and not callable(transform):
            raise TypeError("VidoreTrainDataset transform must be a callable SampleTransform or None")
        effective_transform = (
            transform
            if transform is not None
            else build_vidore_train_default_transform(
                dataset_name=dataset_name,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=lance_path,
            read_columns=read_columns,
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path or str(Path(lance_path).parents[1])
        self.dataset_name = dataset_name
        self.split = "train"
        self.subset = None
        self.variant = None

    @classmethod
    def resolve_train_spec(cls, config: Mapping[str, Any]) -> ResolvedTrainDatasetSpec:
        path = str(config["path"])
        dataset_name = str(config.get("dataset_name", "vidore_train"))
        metadata = dict(config.get("metadata") or {})
        metadata.setdefault("dataset_name", dataset_name)
        metadata.setdefault("split", "train")
        return ResolvedTrainDatasetSpec(
            dataset_type=str(config.get("type", "vidore_train")),
            dataset_name=dataset_name,
            primary_path=f"{path}/data/train.lance",
            lookup_paths={},
            split="train",
            metadata=metadata,
            init_kwargs={
                key: value
                for key, value in config.items()
                if key not in {"type", "path", "dataset_name", "metadata"}
            },
        )

    @classmethod
    def from_resolved_spec(cls, spec: ResolvedTrainDatasetSpec) -> VidoreTrainDataset:
        init_kwargs = dict(spec.init_kwargs)
        init_kwargs.setdefault("metadata", spec.metadata)
        return cls(
            path=init_kwargs.pop("path", None),
            lance_path=spec.primary_path,
            dataset_name=spec.dataset_name or "vidore_train",
            **init_kwargs,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VidoreTrainDataset:
        """Build a ViDoRe full-view dataset from an AutoDataset config mapping."""

        normalized_config = dict(config)
        transform_config = normalized_config.pop("transform", None)
        if transform_config is not None:
            if callable(transform_config):
                normalized_config["transform"] = transform_config
            else:
                normalized_config["transform_kwargs"] = transform_config
        spec = cls.resolve_train_spec(normalized_config)
        return cls.from_resolved_spec(spec)


__all__ = [
    "DATASET_ARCHETYPE",
    "VidoreTrainDataset",
    "build_vidore_full_query_text",
    "build_vidore_image_media",
    "build_vidore_train_default_transform",
    "transform_vidore_answer",
    "transform_vidore_train_sample",
]
