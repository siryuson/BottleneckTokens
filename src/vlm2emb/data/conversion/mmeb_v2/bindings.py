"""Resolve MMEB-V2 conversion inputs from local roots or HF dataset identities.

This module separates two source categories used by conversion:

- local sources under explicit ``MMEB-V2`` or ``MMEB-eval`` roots
- online HuggingFace dataset references used for selected annotations/fixes

It also provides thin readers for local parquet/jsonl assets and HF-loaded
records so pipeline modules can consume normalized inputs without duplicating
binding logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
from datasets import Dataset, concatenate_datasets, load_dataset
from pyarrow.parquet import read_table

from .datasets import DatasetSpec, SourceSpec


def _require_root(root: Path | None, *, arg_name: str, source: SourceSpec) -> Path:
    """Require one explicit CLI-provided root for one source."""

    if root is None:
        raise ValueError(
            f"Missing required root '{arg_name}' for source {source.key!r}. "
            "Pass the root explicitly through the MMEB-V2 CLI or pipeline call."
        )
    return root


def get_source_by_role(spec: DatasetSpec, role: str) -> SourceSpec | None:
    """Return the first source with the requested role from one dataset entry."""

    for source in spec.sources:
        if source.role == role:
            return source
    return None


def parse_hf_dataset_ref(ref: str) -> tuple[str, str | None, str]:
    """Parse one HF dataset identity in ``repo::subset/split`` form."""

    if "::" not in ref:
        raise ValueError(f"Invalid hf dataset ref: {ref}")
    repo, split_ref = ref.split("::", 1)
    if "/" in split_ref:
        subset, split = split_ref.rsplit("/", 1)
        return repo, subset, split
    return repo, None, split_ref


def resolve_source_path(
    source: SourceSpec,
    *,
    mmeb_v2_root: Path | None = None,
    mmeb_eval_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> Path:
    """Resolve one local source from explicit roots and optional exact overrides."""

    if source_overrides and source.key in source_overrides:
        return source_overrides[source.key]

    if source.root == "mmeb_v2":
        root = _require_root(mmeb_v2_root, arg_name="mmeb_v2_root", source=source)
        return root / source.ref
    if source.root == "mmeb_eval":
        root = _require_root(mmeb_eval_root, arg_name="mmeb_eval_root", source=source)
        return root / source.ref
    if source.root == "hf":
        raise ValueError(
            f"HF source {source.key!r} is not a local path source; "
            "load it through HuggingFace datasets instead."
        )
    raise ValueError(f"Unsupported MMEB-V2 source root: {source.root!r}")


def load_hf_records(
    ref: str,
    *,
    subset_names: tuple[str, ...] | None = None,
    subset_field_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load HF rows directly from one repo identity."""

    repo, subset, split = parse_hf_dataset_ref(ref)

    if subset_names is not None:
        datasets: list[Dataset] = []
        for subset_name in subset_names:
            dataset = cast(Dataset, load_dataset(repo, subset_name, split=split))
            if subset_field_name is not None:
                dataset = cast(
                    Dataset,
                    cast(Any, dataset).add_column(
                        subset_field_name,
                        [subset_name] * len(dataset),
                    ),
                )
            datasets.append(dataset)
        merged = concatenate_datasets(datasets)
        return cast(Dataset, merged).to_list()

    if subset is not None:
        return cast(Dataset, load_dataset(repo, subset, split=split)).to_list()
    return cast(Dataset, load_dataset(repo, split=split)).to_list()


def load_hf_beir_table(ref: str, table_name: str) -> list[dict[str, Any]]:
    """Load one BEIR-style table directly from HF."""

    repo, _subset, split = parse_hf_dataset_ref(ref)
    return cast(Dataset, load_dataset(repo, table_name, split=split)).to_list()


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file into a list of row dicts."""

    records: list[dict[str, Any]] = []
    with path.open() as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def read_parquet_shards(
    dataset_dir: Path,
    *,
    patterns: tuple[str, ...] = ("test-*.parquet",),
) -> pa.Table:
    """Read all parquet shards matching the first successful glob pattern."""

    shard_paths: list[Path] = []
    for pattern in patterns:
        shard_paths = sorted(dataset_dir.glob(pattern))
        if shard_paths:
            break
    if not shard_paths:
        raise FileNotFoundError(
            f"No parquet files found in {dataset_dir} for patterns {list(patterns)}"
        )

    tables = [read_table(shard_path) for shard_path in shard_paths]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]
