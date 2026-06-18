"""ScienceQA training conversion utilities."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

SPLIT_SCHEMA = pa.schema(
    [
        pa.field("image_path", pa.string()),
        pa.field("question", pa.string()),
        pa.field("choices", pa.list_(pa.string())),
        pa.field("answer", pa.int32()),
        pa.field("hint", pa.string()),
        pa.field("task", pa.string()),
        pa.field("grade", pa.string()),
        pa.field("subject", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("category", pa.string()),
        pa.field("skill", pa.string()),
        pa.field("lecture", pa.string()),
        pa.field("solution", pa.string()),
    ]
)
IMAGE_SCHEMA = pa.schema([pa.field("path", pa.string()), pa.field("image", pa.binary())])
SUPPORTED_SPLITS = ("official_train", "official_validation", "official_test", "official_train_without_mmeb_v2_eval")


def _batched(items: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield fixed-size row batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _split_files(raw_root: Path, split: str) -> list[Path]:
    """Return sorted ScienceQA parquet files for one split."""

    files = sorted(raw_root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing ScienceQA split files: {raw_root}/{split}-*.parquet")
    return files


def _iter_raw_rows(raw_root: Path, split: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield raw parquet rows with stable split-local row indices."""

    row_index = 0
    for path in _split_files(raw_root, split):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            for row in pa.Table.from_batches([batch]).to_pylist():
                yield row_index, row
                row_index += 1


def _image_payload(row: dict[str, Any]) -> bytes | None:
    """Extract embedded image bytes from one raw row."""

    image = row.get("image")
    if image is None:
        return None
    if isinstance(image, dict):
        payload = image.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
    return None


def _split_row(split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    """Build one raw-preserving ScienceQA split row."""

    image_path = f"{split}/{row_index}" if _image_payload(row) is not None else None
    return {
        "image_path": image_path,
        "question": str(row.get("question", "") or ""),
        "choices": [str(choice) for choice in (row.get("choices") or [])],
        "answer": int(row.get("answer")),
        "hint": row.get("hint"),
        "task": row.get("task"),
        "grade": row.get("grade"),
        "subject": row.get("subject"),
        "topic": row.get("topic"),
        "category": row.get("category"),
        "skill": row.get("skill"),
        "lecture": row.get("lecture"),
        "solution": row.get("solution"),
    }


def _split_batches(raw_root: Path, split: str, *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield ScienceQA split rows as Arrow batches."""

    rows = (_split_row(split, row_index, row) for row_index, row in _iter_raw_rows(raw_root, split))
    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SPLIT_SCHEMA)


def _image_batches(raw_root: Path, *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield ScienceQA images from all official splits."""

    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for row_index, row in _iter_raw_rows(raw_root, split):
            payload = _image_payload(row)
            if payload is None:
                continue
            rows.append({"path": f"{split}/{row_index}", "image": payload})
            if len(rows) >= batch_size:
                yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)
                rows = []
    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_scienceqa_train_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the ScienceQA runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing ScienceQA image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_scienceqa_train_root(
    *,
    raw_root: Path,
    output_root: Path,
    overwrite: bool = False,
    split_batch_size: int = 4096,
    image_batch_size: int = 1024,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert ScienceQA into raw-preserving Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    split_map = {
        "official_train": "train",
        "official_validation": "validation",
        "official_test": "test",
        "official_train_without_mmeb_v2_eval": "train",
    }
    split_counts: dict[str, int] = {}
    for output_split, raw_split in split_map.items():
        path = output_root / "data" / f"{output_split}.lance"
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                SPLIT_SCHEMA,
                _split_batches(raw_root, raw_split, batch_size=split_batch_size),
            ),
            str(path),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
        split_counts[output_split] = lance.dataset(str(path)).count_rows()

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches(raw_root, "test", batch_size=split_batch_size),
        ),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            IMAGE_SCHEMA,
            _image_batches(raw_root, batch_size=image_batch_size),
        ),
        str(output_root / "data" / "images.lance"),
        schema=IMAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_scienceqa_train_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "scienceqa_train_raw_lance_v1",
        "splits": split_counts,
        "exclusion_rows": split_counts["official_test"],
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
    }


__all__ = [
    "IMAGE_SCHEMA",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "ensure_scienceqa_train_indices",
    "write_scienceqa_train_root",
]
