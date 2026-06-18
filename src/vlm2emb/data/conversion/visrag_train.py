"""VisRAG raw-preserving Lance conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

VISRAG_SUBSET_SOURCES: tuple[str, ...] = (
    "InfoVQA",
    "ArxivQA",
    "PlotQA",
    "SlideVQA",
    "MP-DocVQA",
    "ChartQA",
)
VISRAG_FULL_ROOT_NAME = "openbmb_VisRAG-Ret-Train-In-domain-data"
VISRAG_SUBSET_ROOT_PREFIX = "visrag_indomain"


@dataclass(frozen=True)
class LanceWriteSummary:
    """Summary for one converted Lance table."""

    output_path: str
    rows: int


@dataclass(frozen=True)
class VisragConversionSummary:
    """Summary for a VisRAG full-view and optional source-specific conversion."""

    full: LanceWriteSummary
    source_subsets: tuple[LanceWriteSummary, ...] = ()


def discover_parquet_files(input_dir: str | Path, *, split: str = "train") -> list[Path]:
    """Discover raw HuggingFace-style parquet files for one split."""

    root = Path(input_dir)
    candidates = sorted((root / "data").glob(f"{split}-*.parquet"))
    if candidates:
        return candidates
    candidates = sorted(root.glob(f"{split}-*.parquet"))
    if candidates:
        return candidates
    direct = root / f"{split}.parquet"
    if direct.is_file():
        return [direct]
    raise FileNotFoundError(f"No VisRAG parquet files found for split={split!r} under {root}")


def _write_dataset_batches(
    batches: list[pa.RecordBatch] | Any,
    output_path: Path,
    *,
    max_bytes_per_file: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    created = False
    for batch in batches:
        if batch.num_rows == 0:
            continue
        mode = "overwrite" if not created else "append"
        lance.write_dataset(
            pa.Table.from_batches([batch]),
            str(output_path),
            mode=mode,
            max_bytes_per_file=max_bytes_per_file,
        )
        created = True
        rows += batch.num_rows
    if not created:
        raise ValueError(f"No rows were written to {output_path}")
    return rows


def _scan_parquet_batches(
    parquet_files: list[Path],
    *,
    batch_size: int,
) -> Any:
    dataset = ds.dataset([str(path) for path in parquet_files], format="parquet")
    return dataset.scanner(batch_size=batch_size).to_batches()


def convert_visrag_full_view(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> LanceWriteSummary:
    """Convert raw VisRAG parquet shards into one raw-preserving Lance table."""

    parquet_files = discover_parquet_files(input_dir, split=split)
    output_path = Path(output_dir) / "data" / f"{split}.lance"
    rows = _write_dataset_batches(
        _scan_parquet_batches(parquet_files, batch_size=batch_size),
        output_path,
        max_bytes_per_file=max_bytes_per_file,
    )
    return LanceWriteSummary(output_path=str(output_path), rows=rows)


def _subset_root_name(source: str) -> str:
    return f"{VISRAG_SUBSET_ROOT_PREFIX}_{source.replace('/', '_')}"


def convert_visrag_source_subsets(
    full_view_dir: str | Path,
    output_base: str | Path,
    *,
    sources: tuple[str, ...] = VISRAG_SUBSET_SOURCES,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> tuple[LanceWriteSummary, ...]:
    """Write source-specific VisRAG tables with one full-view scan."""

    input_path = Path(full_view_dir) / "data" / f"{split}.lance"
    dataset = lance.dataset(str(input_path))
    base = Path(output_base)
    output_paths = {
        source: base / _subset_root_name(source) / "data" / f"{split}.lance"
        for source in sources
    }
    created = {source: False for source in sources}
    rows = {source: 0 for source in sources}
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for batch in dataset.scanner(batch_size=batch_size, batch_readahead=2).to_batches():
        source_column = batch.column("source")
        for source in sources:
            filtered = batch.filter(pc.equal(source_column, pa.scalar(source)))
            if filtered.num_rows == 0:
                continue
            mode = "overwrite" if not created[source] else "append"
            lance.write_dataset(
                pa.Table.from_batches([filtered]),
                str(output_paths[source]),
                mode=mode,
                max_bytes_per_file=max_bytes_per_file,
            )
            created[source] = True
            rows[source] += filtered.num_rows

    missing_sources = [source for source, was_created in created.items() if not was_created]
    if missing_sources:
        raise ValueError(
            "No rows were written for VisRAG source(s): "
            + ", ".join(missing_sources)
        )

    return tuple(
        LanceWriteSummary(output_path=str(output_paths[source]), rows=rows[source])
        for source in sources
    )


def convert_visrag_root(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    source_output_base: str | Path | None = None,
    sources: tuple[str, ...] = VISRAG_SUBSET_SOURCES,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> VisragConversionSummary:
    """Convert VisRAG full view and optionally source-specific views."""

    full = convert_visrag_full_view(
        input_dir,
        output_dir,
        split=split,
        batch_size=batch_size,
        max_bytes_per_file=max_bytes_per_file,
    )
    source_summaries: tuple[LanceWriteSummary, ...] = ()
    if source_output_base is not None:
        source_summaries = convert_visrag_source_subsets(
            output_dir,
            source_output_base,
            sources=sources,
            split=split,
            batch_size=batch_size,
            max_bytes_per_file=max_bytes_per_file,
        )
    return VisragConversionSummary(full=full, source_subsets=source_summaries)


__all__ = [
    "LANCE_MAX_BYTES_PER_FILE",
    "LanceWriteSummary",
    "VISRAG_FULL_ROOT_NAME",
    "VISRAG_SUBSET_ROOT_PREFIX",
    "VISRAG_SUBSET_SOURCES",
    "VisragConversionSummary",
    "convert_visrag_full_view",
    "convert_visrag_root",
    "convert_visrag_source_subsets",
    "discover_parquet_files",
]
