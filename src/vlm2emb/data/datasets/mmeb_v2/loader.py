"""MMEB-V2 eval dataset loader.

This module builds one retrieval-eval dataset from one converted MMEB-V2
subset artifact root. Parser resolution itself lives in ``dispatch.py`` so the
dataset builder stays separate from the static dataset-to-parser map.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import lance

from vlm2emb.data.datasets.base import EvalLanceDataset, LanceDataset
from vlm2emb.data.datasets.eval import RetrievalEvalDataset
from vlm2emb.data.datasets.mmeb_v2.dispatch import get_parser_module


def _resolve_declared_columns(
    lance_path: str,
    *,
    declared: tuple[str, ...],
) -> list[str]:
    """Validate parser-declared read columns against the raw Lance schema."""

    available_columns = set(lance.dataset(lance_path).schema.names)
    missing = [column for column in declared if column not in available_columns]
    if missing:
        raise ValueError(
            f"Missing required eval columns for {lance_path}: {', '.join(missing)}"
        )
    return list(declared)


def build_mmeb_eval_dataset(
    *,
    dataset_name: str,
    artifact_path: str,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> RetrievalEvalDataset:
    """Build one MMEB retrieval eval dataset from one subset artifact root.

    The loader deliberately performs three explicit steps instead of relying on
    a shared heuristic eval loader:

    1. resolve the dataset-local parser module from the subset name;
    2. ask that parser which raw columns are required on the query/candidate
       side and fail fast if the artifact no longer matches the contract;
    3. compose `EvalLanceDataset + parser transform` so benchmark-specific
       prompt/layout logic stays below the loader boundary.

    MMEB-V2 eval uses explicit query/candidate transform builders. Keep this
    boundary intact so text formatting stays inside parser-owned transforms.
    """

    queries_path = os.path.join(artifact_path, "queries.lance")
    candidates_path = os.path.join(artifact_path, "candidates.lance")
    qrels_path = os.path.join(artifact_path, "qrels.lance")

    parser_module = get_parser_module(dataset_name)
    effective_transform_kwargs = dict(parser_module.get_default_transform_kwargs(dataset_name))
    if transform_kwargs:
        effective_transform_kwargs.update(dict(transform_kwargs))
    query_columns = tuple(parser_module.get_query_read_columns(dataset_name))
    candidate_columns = tuple(parser_module.get_candidate_read_columns(dataset_name))
    query_transform = cast(
        Any,
        parser_module.build_query_transform(
            dataset_name=dataset_name,
            transform_kwargs=effective_transform_kwargs,
        ),
    )
    candidate_transform = cast(
        Any,
        parser_module.build_candidate_transform(
            dataset_name=dataset_name,
            transform_kwargs=effective_transform_kwargs,
        ),
    )

    queries = EvalLanceDataset(
        queries_path,
        read_columns=_resolve_declared_columns(
            queries_path,
            declared=query_columns,
        ),
        transform=query_transform,
    )
    candidates = EvalLanceDataset(
        candidates_path,
        read_columns=_resolve_declared_columns(
            candidates_path,
            declared=candidate_columns,
        ),
        transform=candidate_transform,
    )
    qrels = LanceDataset(qrels_path)

    return RetrievalEvalDataset(
        queries=queries,
        candidates=candidates,
        qrels=qrels,
        metadata={
            "name": dataset_name,
            "benchmark": "MMEB-V2",
            "runtime_mode": effective_transform_kwargs.get("runtime_mode", "canonical"),
        },
    )


__all__ = ["build_mmeb_eval_dataset"]
