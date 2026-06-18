"""Convert MMEB-V2 visdoc evaluation datasets into Lance retrieval artifacts.

The visdoc pipelines consume BEIR-like query/corpus/qrels inputs from local
MMEB-V2 data or approved HuggingFace fixes, flatten the corpus rows into a
stable candidate shape, validate references, and write normalized retrieval
artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from .bindings import (
    get_source_by_role,
    load_hf_beir_table,
    read_parquet_shards,
    resolve_source_path,
)
from .common import ensure_output_root, get_metadata_task_type, write_minimal_metadata
from .datasets import VISDOC_MODALITY, get_modality, get_pipeline, get_spec, get_specs_by_modality

BEIR_PARQUET_PATTERNS = ("test-*.parquet", "train-*.parquet")


def _write_lance(records: list[dict[str, Any]], path: Path) -> None:
    """Write one record list into a Lance dataset with inferred schema."""
    table = pa.Table.from_pylist(records)
    lance.write_dataset(table, str(path), mode="create")


def _read_local_beir_table(data_dir: Path, table_name: str) -> pa.Table:
    """Read one local BEIR parquet table from one visdoc dataset directory."""
    return read_parquet_shards(data_dir / table_name, patterns=BEIR_PARQUET_PATTERNS)


def _resolve_visdoc_local_data_dir(
    dataset_name: str,
    *,
    data_dir: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> Path:
    """Resolve one local visdoc BEIR directory from explicit inputs."""
    if data_dir is not None:
        return data_dir

    spec = get_spec(dataset_name)
    annotation_source = get_source_by_role(spec, "annotation")
    if annotation_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing annotation source in dataset list")

    return Path(
        resolve_source_path(
            annotation_source,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
        )
    )


def _load_beir_tables(
    dataset_name: str,
    *,
    data_dir: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Load BEIR-style queries/corpus/qrels tables from local files or HF."""
    spec = get_spec(dataset_name)
    annotation_source = get_source_by_role(spec, "annotation")
    if annotation_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing annotation source in dataset list")

    if annotation_source.kind == "hf_dataset":
        return (
            pa.Table.from_pylist(load_hf_beir_table(annotation_source.ref, "queries")),
            pa.Table.from_pylist(load_hf_beir_table(annotation_source.ref, "corpus")),
            pa.Table.from_pylist(load_hf_beir_table(annotation_source.ref, "qrels")),
        )

    dataset_path = _resolve_visdoc_local_data_dir(
        dataset_name,
        data_dir=data_dir,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    return (
        _read_local_beir_table(dataset_path, "queries"),
        _read_local_beir_table(dataset_path, "corpus"),
        _read_local_beir_table(dataset_path, "qrels"),
    )


def _flatten_corpus_row(dataset_name: str, row: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw BEIR corpus row into the Lance candidate row shape."""
    corpus_id = row.get("corpus-id")
    image_struct = row.get("image")
    if not isinstance(image_struct, dict):
        raise ValueError(f"[{dataset_name}] corpus.image must be a struct, got {type(image_struct).__name__}")

    image_bytes = image_struct.get("bytes")
    if image_bytes is None:
        raise ValueError(f"[{dataset_name}] corpus.image.bytes is missing for corpus-id={corpus_id!r}")

    flattened = {"id": corpus_id, "image": image_bytes, "path": image_struct.get("path")}
    for key, value in row.items():
        if key in {"corpus-id", "image"}:
            continue
        flattened[key] = value
    return flattened


def _aggregate_qrels(
    queries_table: pa.Table,
    corpus_table: pa.Table,
    qrels_table: pa.Table,
    *,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], dict[Any, dict[str, list[Any]]]]:
    """Aggregate raw BEIR qrels into one row per query with sparse judgments."""
    standard_columns = {"query-id", "corpus-id", "score"}
    extra_columns = [name for name in qrels_table.column_names if name not in standard_columns]
    query_id_list = queries_table["query-id"].to_pylist()
    corpus_id_list = corpus_table["corpus-id"].to_pylist()
    if any(query_id is None for query_id in query_id_list):
        raise ValueError(f"[{dataset_name}] queries contains null query-id values")
    if any(corpus_id is None for corpus_id in corpus_id_list):
        raise ValueError(f"[{dataset_name}] corpus contains null corpus-id values")
    query_ids = set(query_id_list)
    corpus_ids = set(corpus_id_list)

    query_id_counts = Counter(query_id_list)
    corpus_id_counts = Counter(corpus_id_list)

    if len(query_ids) != len(query_id_list):
        duplicate_query_ids = sorted(
            [query_id for query_id, count in query_id_counts.items() if count > 1],
            key=repr,
        )
        raise ValueError(
            f"[{dataset_name}] queries contains duplicate query-id values: {duplicate_query_ids!r}"
        )
    if len(corpus_ids) != len(corpus_id_list):
        duplicate_corpus_ids = sorted(
            [corpus_id for corpus_id, count in corpus_id_counts.items() if count > 1],
            key=repr,
        )
        raise ValueError(
            f"[{dataset_name}] corpus contains duplicate corpus-id values: {duplicate_corpus_ids!r}"
        )

    grouped_candidate_ids: dict[Any, list[Any]] = defaultdict(list)
    grouped_candidate_scores: dict[Any, list[float]] = defaultdict(list)
    grouped_extras: dict[Any, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))

    for row in qrels_table.to_pylist():
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        if query_id not in query_ids:
            raise ValueError(
                f"[{dataset_name}] qrels references unknown query-id: {query_id!r}"
            )
        if corpus_id not in corpus_ids:
            raise ValueError(
                f"[{dataset_name}] qrels references unknown corpus-id: {corpus_id!r}"
            )
        grouped_candidate_ids[query_id].append(row["corpus-id"])
        grouped_candidate_scores[query_id].append(float(row["score"]))
        for field_name in extra_columns:
            grouped_extras[query_id][field_name].append(row[field_name])

    missing_query_ids = [
        query_id
        for query_id in query_id_list
        if query_id not in grouped_candidate_ids
    ]
    if missing_query_ids:
        raise ValueError(
            f"[{dataset_name}] qrels missing entries for query-id values: {missing_query_ids!r}"
        )

    qrels: list[dict[str, Any]] = []
    for query_id in query_id_list:
        qrels.append(
            {
                "query_id": query_id,
                "mode": "sparse",
                "candidate_ids": grouped_candidate_ids.get(query_id, []),
                "candidate_scores": grouped_candidate_scores.get(query_id, []),
            }
        )

    return qrels, grouped_extras


def _build_query_rows(
    queries_table: pa.Table,
    *,
    qrels_extras: dict[Any, dict[str, list[Any]]],
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Build query rows and merge qrels-side extra columns when present."""
    rows: list[dict[str, Any]] = []
    for row in queries_table.to_pylist():
        query_id = row["query-id"]
        query_row: dict[str, Any] = {"id": query_id}
        for key, value in row.items():
            if key == "query-id":
                continue
            query_row[key] = value

        for extra_name, values in qrels_extras.get(query_id, {}).items():
            if extra_name in query_row:
                raise ValueError(
                    f"[{dataset_name}] query field collision while merging qrels extras: {extra_name!r}"
                )
            query_row[extra_name] = values

        rows.append(query_row)
    return rows


def convert_visdoc_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    data_dir: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> Path:
    """Convert one MMEB-V2 visdoc dataset into rerun layout."""

    if get_modality(dataset_name) != VISDOC_MODALITY:
        raise ValueError(f"{dataset_name!r} is not a visdoc dataset")

    queries_table, corpus_table, qrels_table = _load_beir_tables(
        dataset_name,
        data_dir=data_dir,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    output_path = ensure_output_root(output_root / "visdoc-tasks" / dataset_name)

    qrels, qrels_extras = _aggregate_qrels(
        queries_table,
        corpus_table,
        qrels_table,
        dataset_name=dataset_name,
    )
    queries = _build_query_rows(
        queries_table,
        qrels_extras=qrels_extras,
        dataset_name=dataset_name,
    )
    candidates = [
        _flatten_corpus_row(dataset_name, row)
        for row in corpus_table.to_pylist()
    ]

    _write_lance(queries, output_path / "queries.lance")
    _write_lance(candidates, output_path / "candidates.lance")
    _write_lance(qrels, output_path / "qrels.lance")
    write_minimal_metadata(
        output_path,
        dataset_name=dataset_name,
        task_type=get_metadata_task_type(dataset_name, get_pipeline(dataset_name)),
        modality=get_modality(dataset_name),
    )
    return output_path


def convert_all_visdoc_datasets(
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> list[Path]:
    """Convert all visdoc datasets declared in the dataset list."""
    outputs: list[Path] = []
    for spec in get_specs_by_modality(VISDOC_MODALITY):
        outputs.append(
            convert_visdoc_dataset(
                spec.name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
            )
        )
    return outputs
