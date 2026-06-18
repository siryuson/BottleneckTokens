"""MMEB-train raw-preserving Lance conversion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

VARIANT_MAP: dict[str, str] = {
    "train": "train-00000-of-00001.parquet",
    "original": "original-00000-of-00001.parquet",
    "diverse_instruction": "diverse_instruction-00000-of-00001.parquet",
}
IMAGE_PATH_COLUMNS: tuple[str, ...] = (
    "qry_image_path",
    "pos_image_path",
    "neg_image_path",
)
IMAGE_SCHEMA = pa.schema([
    pa.field("path", pa.string()),
    pa.field("image", pa.binary()),
])
SAMPLE_SCHEMA: pa.Schema | None = None
DEFAULT_NUM_WORKERS = 16
DEFAULT_IMAGE_BATCH_SIZE = 2048
MMEB_TRAIN_EMPTY_SIDE_CLEANUP_SUBSETS = frozenset({"A-OKVQA", "OK-VQA"})
_EMPTY_SIDE_COLUMNS = ("qry", "qry_image_path", "pos_text", "pos_image_path")


@dataclass(frozen=True)
class SplitSummary:
    """Summary for one converted raw parquet split."""

    name: str
    source_path: str
    output_path: str
    rows: int
    dropped_rows: int = 0


@dataclass(frozen=True)
class DatasetSummary:
    """Summary for one converted MMEB-train subset."""

    subset: str
    splits: tuple[SplitSummary, ...]
    image_output_path: str
    image_rows: int


@dataclass(frozen=True)
class ImageReference:
    """One raw image path plus source context for diagnostics."""

    path: str
    subset: str
    split: str
    row_index: int
    column: str


def discover_subsets(input_dir: str | Path) -> list[str]:
    """Discover MMEB-train subset directories with at least one known split."""

    root = Path(input_dir)
    return [
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and any((path / filename).is_file() for filename in VARIANT_MAP.values())
    ]


def _variant_paths(input_dir: Path, subset: str) -> list[tuple[str, Path]]:
    subset_dir = input_dir / subset
    return [
        (split_name, subset_dir / filename)
        for split_name, filename in VARIANT_MAP.items()
        if (subset_dir / filename).is_file()
    ]


def _write_lance_table(
    table: pa.Table,
    output_path: Path,
    *,
    schema: pa.Schema | None = None,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(
        table,
        str(output_path),
        schema=schema,
        mode="overwrite",
        max_bytes_per_file=max_bytes_per_file,
    )


def _iter_image_references(
    table: pa.Table,
    *,
    subset: str,
    split: str,
) -> list[ImageReference]:
    columns = [name for name in IMAGE_PATH_COLUMNS if name in table.column_names]
    if not columns:
        return []
    rows = table.select(columns).to_pylist()
    references: list[ImageReference] = []
    for row_index, row in enumerate(rows):
        for column in columns:
            raw_path = row.get(column)
            if raw_path is None:
                continue
            image_path = str(raw_path).strip()
            if not image_path:
                continue
            references.append(
                ImageReference(
                    path=image_path,
                    subset=subset,
                    split=split,
                    row_index=row_index,
                    column=column,
                )
            )
    return references


def _read_image(reference: ImageReference, image_root: Path) -> tuple[str, bytes]:
    image_path = image_root / reference.path
    if not image_path.is_file():
        _raise_missing_image(reference, image_path)
    return reference.path, image_path.read_bytes()


def _raise_missing_image(reference: ImageReference, image_path: Path) -> None:
    raise FileNotFoundError(
        "MMEB-train image asset is missing: "
        f"dataset={reference.subset}, split={reference.split}, row={reference.row_index}, "
        f"column={reference.column}, image_path={reference.path}, resolved_path={image_path}"
    )


def _check_image_reference_exists(reference: ImageReference, image_root: Path) -> None:
    image_path = image_root / reference.path
    if not image_path.is_file():
        _raise_missing_image(reference, image_path)


def _check_image_references_exist(
    references: list[ImageReference],
    image_root: Path,
    *,
    num_workers: int,
) -> None:
    worker_count = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_check_image_reference_exists, reference, image_root)
            for reference in references
        ]
        for future in futures:
            future.result()


def _dedupe_references(references: list[ImageReference]) -> list[ImageReference]:
    first_by_path: dict[str, ImageReference] = {}
    for reference in references:
        first_by_path.setdefault(reference.path, reference)
    return [first_by_path[path] for path in sorted(first_by_path)]


def _read_image_batch(
    references: list[ImageReference],
    *,
    image_root: Path,
    num_workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    worker_count = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_read_image, reference, image_root) for reference in references]
        for future in as_completed(futures):
            path, image = future.result()
            rows.append({"path": path, "image": image})
    rows.sort(key=lambda row: row["path"])
    return rows


def _iter_batches(values: list[ImageReference], batch_size: int) -> list[list[ImageReference]]:
    normalized_batch_size = max(1, int(batch_size))
    return [
        values[start : start + normalized_batch_size]
        for start in range(0, len(values), normalized_batch_size)
    ]


def _is_empty_raw_value(value: Any) -> bool:
    return value is None or value == ""


def _find_empty_side_row_indices(table: pa.Table) -> list[int]:
    """Find rows that the origin VLM2Vec MMEB parser would skip."""

    if any(column not in table.column_names for column in _EMPTY_SIDE_COLUMNS):
        return []
    rows = table.select(_EMPTY_SIDE_COLUMNS).to_pylist()
    invalid_indices: list[int] = []
    for row_index, row in enumerate(rows):
        empty_query = _is_empty_raw_value(row["qry"]) and _is_empty_raw_value(
            row["qry_image_path"]
        )
        empty_positive = _is_empty_raw_value(row["pos_text"]) and _is_empty_raw_value(
            row["pos_image_path"]
        )
        if empty_query or empty_positive:
            invalid_indices.append(row_index)
    return invalid_indices


def _drop_empty_side_rows(table: pa.Table, row_indices: list[int]) -> pa.Table:
    if not row_indices:
        return table
    invalid_set = set(row_indices)
    keep_mask = [row_index not in invalid_set for row_index in range(table.num_rows)]
    return table.filter(pa.array(keep_mask, type=pa.bool_()))


def _clean_mmeb_train_split(
    table: pa.Table,
    *,
    subset: str,
    split: str,
    source_path: Path,
) -> tuple[pa.Table, int]:
    """Drop known bad MMEB QA rows without hiding new data issues."""

    invalid_indices = _find_empty_side_row_indices(table)
    if not invalid_indices:
        return table, 0
    if subset not in MMEB_TRAIN_EMPTY_SIDE_CLEANUP_SUBSETS:
        preview = ", ".join(str(index) for index in invalid_indices[:10])
        raise ValueError(
            "MMEB-train split contains empty query or positive side outside "
            f"the narrow cleanup allowlist: subset={subset}, split={split}, "
            f"source_path={source_path}, row_indices=[{preview}]"
        )
    return _drop_empty_side_rows(table, invalid_indices), len(invalid_indices)


def convert_images(
    *,
    image_root: str | Path,
    output_path: str | Path,
    references: list[ImageReference],
    num_workers: int = DEFAULT_NUM_WORKERS,
    batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> int:
    """Write a MMEB image side table using `path` and `image` fields."""

    output = Path(output_path)
    unique_references = _dedupe_references(references)
    output.parent.mkdir(parents=True, exist_ok=True)
    if unique_references:
        root = Path(image_root)
        _check_image_references_exist(unique_references, root, num_workers=num_workers)

        def generate_batches() -> Any:
            for reference_batch in _iter_batches(unique_references, batch_size):
                rows = _read_image_batch(
                    reference_batch,
                    image_root=root,
                    num_workers=num_workers,
                )
                yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)

        reader = pa.RecordBatchReader.from_batches(IMAGE_SCHEMA, generate_batches())
        lance.write_dataset(
            reader,
            str(output),
            schema=IMAGE_SCHEMA,
            mode="overwrite",
            max_bytes_per_file=max_bytes_per_file,
        )
    else:
        table = pa.Table.from_pylist([], schema=IMAGE_SCHEMA)
        _write_lance_table(
            table,
            output,
            schema=IMAGE_SCHEMA,
            max_bytes_per_file=max_bytes_per_file,
        )
    try:
        lance.dataset(str(output)).create_scalar_index("path", "BTREE")
    except Exception:
        # Index creation is an optimization; conversion output remains valid if
        # Lance skips it for an empty table or an existing index.
        pass
    return len(unique_references)


def convert_metadata(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    *,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> tuple[tuple[SplitSummary, ...], list[ImageReference]]:
    """Convert one subset's raw parquet split tables without adding columns."""

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    split_summaries: list[SplitSummary] = []
    image_references: list[ImageReference] = []
    for split_name, source_path in _variant_paths(input_root, dataset_name):
        table = pq.read_table(source_path)
        table, dropped_rows = _clean_mmeb_train_split(
            table,
            subset=dataset_name,
            split=split_name,
            source_path=source_path,
        )
        output_path = output_root / "data" / dataset_name / f"{split_name}.lance"
        _write_lance_table(table, output_path, max_bytes_per_file=max_bytes_per_file)
        split_summaries.append(
            SplitSummary(
                name=split_name,
                source_path=str(source_path),
                output_path=str(output_path),
                rows=table.num_rows,
                dropped_rows=dropped_rows,
            )
        )
        image_references.extend(
            _iter_image_references(table, subset=dataset_name, split=split_name)
        )
    if not split_summaries:
        raise FileNotFoundError(f"No MMEB-train parquet splits found for subset={dataset_name}")
    return tuple(split_summaries), image_references


def convert_mmeb_train_subset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    subset: str,
    image_root: str | Path | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> DatasetSummary:
    """Convert one MMEB-train subset to raw sample tables plus image side table."""

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    split_summaries, references = convert_metadata(
        input_root,
        output_root,
        subset,
        max_bytes_per_file=max_bytes_per_file,
    )
    image_output_path = output_root / "data" / "images" / f"{subset}.lance"
    image_rows = convert_images(
        image_root=image_root or input_root,
        output_path=image_output_path,
        references=references,
        num_workers=num_workers,
        batch_size=image_batch_size,
        max_bytes_per_file=max_bytes_per_file,
    )
    return DatasetSummary(
        subset=subset,
        splits=split_summaries,
        image_output_path=str(image_output_path),
        image_rows=image_rows,
    )


def convert_mmeb_train_root(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    subsets: list[str] | None = None,
    image_root: str | Path | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> list[DatasetSummary]:
    """Convert a MMEB-train root without writing manifest or inventory files."""

    input_root = Path(input_dir)
    selected_subsets = list(subsets) if subsets else discover_subsets(input_root)
    return [
        convert_mmeb_train_subset(
            input_dir=input_root,
            output_dir=output_dir,
            subset=subset,
            image_root=image_root or input_root,
            num_workers=num_workers,
            image_batch_size=image_batch_size,
            max_bytes_per_file=max_bytes_per_file,
        )
        for subset in selected_subsets
    ]


__all__ = [
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_IMAGE_BATCH_SIZE",
    "IMAGE_PATH_COLUMNS",
    "IMAGE_SCHEMA",
    "LANCE_MAX_BYTES_PER_FILE",
    "MMEB_TRAIN_EMPTY_SIDE_CLEANUP_SUBSETS",
    "SAMPLE_SCHEMA",
    "VARIANT_MAP",
    "DatasetSummary",
    "ImageReference",
    "SplitSummary",
    "convert_images",
    "convert_metadata",
    "convert_mmeb_train_root",
    "convert_mmeb_train_subset",
    "discover_subsets",
]
