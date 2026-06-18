"""Place365 training conversion utilities."""

from __future__ import annotations

import shutil
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

SUPPORTED_SPLITS = ("official_train", "official_val", "official_train_without_mmeb_v2_eval")

SPLIT_SCHEMA = pa.schema(
    [
        pa.field("image_path", pa.string()),
        pa.field("class_name", pa.string()),
    ]
)
IMAGES_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("image", pa.binary()),
    ]
)


def _batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield fixed-size batches while preserving input order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _read_split_file(path: Path, split_dir: str) -> list[dict[str, str]]:
    """Read one Places365 split file into raw-preserving rows."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing Place365 split file: {path}")
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        image_path = line.strip()
        if not image_path:
            continue
        parts = Path(image_path).parts
        if len(parts) < 3 or parts[0] != split_dir:
            raise ValueError(f"Unexpected Place365 split row at {path}:{line_number}: {image_path!r}")
        rows.append({"image_path": image_path, "class_name": parts[1]})
    return rows


def load_place365_split_rows(raw_root: Path) -> OrderedDict[str, list[dict[str, str]]]:
    """Return official train/val rows plus the eval-excluded train view."""

    train = _read_split_file(raw_root / "train.txt", "train")
    val = _read_split_file(raw_root / "val.txt", "val")
    val_paths = {row["image_path"] for row in val}
    train_without_eval = [row for row in train if row["image_path"] not in val_paths]
    return OrderedDict(
        [
            ("official_train", train),
            ("official_val", val),
            ("official_train_without_mmeb_v2_eval", train_without_eval),
        ]
    )


def collect_image_paths(raw_root: Path) -> tuple[OrderedDict[str, None], dict[str, int]]:
    """Collect unique image paths and split row counts without reading bytes."""

    image_paths: OrderedDict[str, None] = OrderedDict()
    split_counts: dict[str, int] = {}
    for split_name, rows in load_place365_split_rows(raw_root).items():
        split_counts[split_name] = len(rows)
        for row in rows:
            image_paths.setdefault(row["image_path"], None)
    return image_paths, split_counts


def _build_image_row(raw_root: Path, image_path: str) -> dict[str, Any]:
    """Read one raw Place365 image row."""

    source_path = raw_root / image_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing Place365 image: {source_path}")
    return {"path": image_path, "image": source_path.read_bytes()}


def _image_batches(
    *,
    raw_root: Path,
    image_paths: OrderedDict[str, None],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build image table batches with bounded parallel file IO."""

    workers = max(1, int(num_workers))
    for chunk in _batched(image_paths.keys(), batch_size):
        if workers == 1:
            rows = [_build_image_row(raw_root, image_path) for image_path in chunk]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(lambda image_path: _build_image_row(raw_root, image_path), chunk))
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGES_SCHEMA)


def _split_batches(rows: Iterable[dict[str, str]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_place365_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Place365 runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing Place365 image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_place365_root(
    *,
    raw_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    image_batch_size: int = 2048,
    split_batch_size: int = 8192,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert Place365 into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    image_paths, split_counts = collect_image_paths(raw_root)
    image_reader = pa.RecordBatchReader.from_batches(
        IMAGES_SCHEMA,
        _image_batches(
            raw_root=raw_root,
            image_paths=image_paths,
            batch_size=image_batch_size,
            num_workers=num_workers,
        ),
    )
    lance.write_dataset(
        image_reader,
        str(output_root / "data" / "images.lance"),
        schema=IMAGES_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_place365_split_rows(raw_root)
    for split_name, rows in split_rows.items():
        reader = pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches(rows, batch_size=split_batch_size),
        )
        lance.write_dataset(
            reader,
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    exclusion_reader = pa.RecordBatchReader.from_batches(
        SPLIT_SCHEMA,
        _split_batches(split_rows["official_val"], batch_size=split_batch_size),
    )
    lance.write_dataset(
        exclusion_reader,
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_place365_indices(output_root)
    return {
        "output_root": str(output_root),
        "image_rows": len(image_paths),
        "splits": split_counts,
        "exclusion_rows": split_counts["official_val"],
        "artifact_format": "place365_train_raw_lance_v1",
    }


__all__ = [
    "IMAGES_SCHEMA",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "collect_image_paths",
    "ensure_place365_indices",
    "load_place365_split_rows",
    "write_place365_root",
]
