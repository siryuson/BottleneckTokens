"""ViDoRe raw-preserving Lance conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

VIDORE_RAG_SUBSET_SOURCES: tuple[str, ...] = (
    "arxiv_qa",
    "pdf",
    "docvqa",
    "tatdqa",
    "Infographic-VQA",
)
VIDORE_FULL_ROOT_NAME = "vidore_colpali_train_set"
VIDORE_RAG_ROOT_PREFIX = "vidore_rag"


@dataclass(frozen=True)
class LanceWriteSummary:
    """Summary for one converted Lance table."""

    output_path: str
    rows: int


@dataclass(frozen=True)
class VidoreConversionSummary:
    """Summary for a ViDoRe full-view and optional RAG-view conversion."""

    full: LanceWriteSummary
    rag_subsets: tuple[LanceWriteSummary, ...] = ()


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
    raise FileNotFoundError(f"No ViDoRe parquet files found for split={split!r} under {root}")


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


def convert_vidore_full_view(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> LanceWriteSummary:
    """Convert raw ViDoRe parquet shards into one raw-preserving Lance table."""

    parquet_files = discover_parquet_files(input_dir, split=split)
    output_path = Path(output_dir) / "data" / f"{split}.lance"
    rows = _write_dataset_batches(
        _scan_parquet_batches(parquet_files, batch_size=batch_size),
        output_path,
        max_bytes_per_file=max_bytes_per_file,
    )
    return LanceWriteSummary(output_path=str(output_path), rows=rows)


def _subset_root_name(source: str) -> str:
    return f"{VIDORE_RAG_ROOT_PREFIX}_{source.replace('/', '_')}"


def convert_vidore_rag_subset(
    full_view_dir: str | Path,
    output_dir: str | Path,
    *,
    source: str,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> LanceWriteSummary:
    """Write one source-specific ViDoRe RAG Lance table from a converted full view."""

    input_path = Path(full_view_dir) / "data" / f"{split}.lance"
    output_path = Path(output_dir) / "data" / f"{split}.lance"
    dataset = lance.dataset(str(input_path))
    rows = _write_dataset_batches(
        (
            batch.filter(pc.equal(batch.column("source"), pa.scalar(source)))
            for batch in dataset.scanner(batch_size=batch_size, batch_readahead=2).to_batches()
        ),
        output_path,
        max_bytes_per_file=max_bytes_per_file,
    )
    return LanceWriteSummary(output_path=str(output_path), rows=rows)


def convert_vidore_rag_subsets(
    full_view_dir: str | Path,
    output_base: str | Path,
    *,
    sources: tuple[str, ...] = VIDORE_RAG_SUBSET_SOURCES,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> tuple[LanceWriteSummary, ...]:
    """Write source-specific ViDoRe RAG tables with one full-view scan."""

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
            "No rows were written for ViDoRe RAG source(s): "
            + ", ".join(missing_sources)
        )

    return tuple(
        LanceWriteSummary(output_path=str(output_paths[source]), rows=rows[source])
        for source in sources
    )


def convert_vidore_root(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    rag_output_base: str | Path | None = None,
    rag_sources: tuple[str, ...] = VIDORE_RAG_SUBSET_SOURCES,
    split: str = "train",
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> VidoreConversionSummary:
    """Convert ViDoRe full view and optionally source-specific RAG views."""

    full = convert_vidore_full_view(
        input_dir,
        output_dir,
        split=split,
        batch_size=batch_size,
        max_bytes_per_file=max_bytes_per_file,
    )
    rag_summaries: list[LanceWriteSummary] = []
    if rag_output_base is not None:
        rag_summaries.extend(
            convert_vidore_rag_subsets(
                output_dir,
                rag_output_base,
                sources=rag_sources,
                split=split,
                batch_size=batch_size,
                max_bytes_per_file=max_bytes_per_file,
            )
        )
    return VidoreConversionSummary(full=full, rag_subsets=tuple(rag_summaries))


__all__ = [
    "LANCE_MAX_BYTES_PER_FILE",
    "LanceWriteSummary",
    "VIDORE_FULL_ROOT_NAME",
    "VIDORE_RAG_ROOT_PREFIX",
    "VIDORE_RAG_SUBSET_SOURCES",
    "VidoreConversionSummary",
    "convert_vidore_full_view",
    "convert_vidore_rag_subset",
    "convert_vidore_rag_subsets",
    "convert_vidore_root",
    "discover_parquet_files",
]
