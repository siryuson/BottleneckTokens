"""GQA training conversion utilities."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

IMAGE_SCHEMA = pa.schema([pa.field("path", pa.string()), pa.field("image", pa.binary())])
SUPPORTED_SPLITS = (
    "official_train_all",
    "official_train_all_without_mmeb_v2_eval",
    "official_train_balanced",
    "official_train_balanced_without_mmeb_v2_eval",
)
DEFAULT_QUERY_PREFIX = "Represent the given image with the following question:"
INSTRUCTION_SUBSETS = ("all", "balanced")


def _parquet_files(root: Path) -> list[Path]:
    """Return sorted parquet files under one directory."""

    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing parquet files under: {root}")
    return files


def _instruction_files(raw_root: Path, subset: str = "all") -> list[Path]:
    """Return GQA instruction parquet files."""

    if subset not in INSTRUCTION_SUBSETS:
        raise ValueError(f"Unsupported GQA instruction subset: {subset}")
    return _parquet_files(raw_root / f"train_{subset}_instructions")


def _image_files(raw_root: Path) -> list[Path]:
    """Return GQA image parquet files."""

    return _parquet_files(raw_root / "train_all_images")


def _instruction_schema(raw_root: Path, subset: str = "all") -> pa.Schema:
    """Read the raw instruction schema without modifying source columns."""

    return pq.ParquetFile(_instruction_files(raw_root, subset=subset)[0]).schema_arrow


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


def _load_image_ids(raw_root: Path) -> set[str]:
    """Load image ids available in the image table source."""

    image_ids: set[str] = set()
    for path in _image_files(raw_root):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["id"]):
            image_ids.update(str(value) for value in batch.column("id").to_pylist())
    return image_ids


def _normalize_eval_question(text: str) -> str:
    """Recover the raw GQA question from a MMEB-V2 query string."""

    normalized = text.replace("<|image_1|>", "").replace("<|image_pad|>", "").strip()
    if normalized.startswith(DEFAULT_QUERY_PREFIX):
        normalized = normalized[len(DEFAULT_QUERY_PREFIX) :].strip()
    return normalized.rstrip("\n")


def load_gqa_eval_overlap_keys(eval_root: Path | None) -> set[tuple[str, str]]:
    """Load MMEB-V2 eval `(imageId, question)` keys for train exclusion."""

    if eval_root is None:
        return set()
    queries_path = eval_root / "queries.lance"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing GQA MMEB-V2 queries table: {queries_path}")
    dataset = lance.dataset(str(queries_path))
    table = dataset.to_table(columns=["qry_img_path", "qry_text"])
    overlap_keys: set[tuple[str, str]] = set()
    for row in table.to_pylist():
        image_path = str(row.get("qry_img_path", "") or "")
        image_id = Path(image_path).stem.removeprefix("image_")
        question = _normalize_eval_question(str(row.get("qry_text", "") or ""))
        if image_id and question:
            overlap_keys.add((image_id, question))
    return overlap_keys


def _instruction_batches(
    raw_root: Path,
    *,
    instruction_subset: str,
    schema: pa.Schema,
    image_ids: set[str],
    eval_overlap_keys: set[tuple[str, str]],
    exclude_eval: bool,
) -> Iterator[pa.RecordBatch]:
    """Yield raw GQA instruction batches after conversion-time row repair."""

    emitted = False
    for path in _instruction_files(raw_root, subset=instruction_subset):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            image_id_values = [str(value) for value in batch.column("imageId").to_pylist()]
            question_values = [str(value) for value in batch.column("question").to_pylist()]
            mask_values = []
            for image_id, question in zip(image_id_values, question_values, strict=True):
                keep = image_id in image_ids
                if exclude_eval and keep:
                    keep = (image_id, question) not in eval_overlap_keys
                mask_values.append(keep)
            filtered = batch.filter(pa.array(mask_values, type=pa.bool_()))
            if filtered.num_rows:
                emitted = True
                yield filtered
    if not emitted:
        yield pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in schema],
            schema=schema,
        )


def _iter_image_rows(raw_root: Path) -> Iterator[dict[str, Any]]:
    """Yield GQA image table rows using image ids as lookup paths."""

    for path in _image_files(raw_root):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["id", "image"]):
            for row in pa.Table.from_batches([batch]).to_pylist():
                image = row.get("image") or {}
                payload = image.get("bytes") if isinstance(image, dict) else None
                if payload is None:
                    continue
                yield {"path": str(row["id"]), "image": bytes(payload)}


def _image_batches(raw_root: Path, *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield GQA image table rows as Arrow batches."""

    for rows in _batched(_iter_image_rows(raw_root), batch_size):
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)


def _write_split(
    *,
    raw_root: Path,
    output_path: Path,
    instruction_subset: str,
    schema: pa.Schema,
    image_ids: set[str],
    eval_overlap_keys: set[tuple[str, str]],
    exclude_eval: bool,
    max_bytes_per_file: int,
) -> int:
    """Write one GQA split table and return its row count."""

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            schema,
            _instruction_batches(
                raw_root,
                instruction_subset=instruction_subset,
                schema=schema,
                image_ids=image_ids,
                eval_overlap_keys=eval_overlap_keys,
                exclude_eval=exclude_eval,
            ),
        ),
        str(output_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    return lance.dataset(str(output_path)).count_rows()


def _write_eval_exclusions(
    *,
    raw_root: Path,
    output_path: Path,
    instruction_subset: str,
    schema: pa.Schema,
    image_ids: set[str],
    eval_overlap_keys: set[tuple[str, str]],
    max_bytes_per_file: int,
) -> int:
    """Write the rows removed from train because they overlap MMEB-V2 eval."""

    if not eval_overlap_keys:
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                schema,
                [pa.RecordBatch.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)],
            ),
            str(output_path),
            schema=schema,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
        return 0

    def batches() -> Iterator[pa.RecordBatch]:
        emitted = False
        for path in _instruction_files(raw_root, subset=instruction_subset):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches():
                image_id_values = [str(value) for value in batch.column("imageId").to_pylist()]
                question_values = [str(value) for value in batch.column("question").to_pylist()]
                mask_values = [
                    image_id in image_ids and (image_id, question) in eval_overlap_keys
                    for image_id, question in zip(image_id_values, question_values, strict=True)
                ]
                filtered = batch.filter(pa.array(mask_values, type=pa.bool_()))
                if filtered.num_rows:
                    emitted = True
                    yield filtered
        if not emitted:
            yield pa.RecordBatch.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, batches()),
        str(output_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    return lance.dataset(str(output_path)).count_rows()


def write_gqa_instruction_splits(
    *,
    raw_root: Path,
    output_root: Path,
    eval_root: Path | None = None,
    overwrite: bool = False,
    instruction_subsets: tuple[str, ...] = INSTRUCTION_SUBSETS,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Write GQA instruction split tables without rewriting the image table."""

    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)
    image_ids = _load_image_ids(raw_root)
    eval_overlap_keys = load_gqa_eval_overlap_keys(eval_root)
    split_counts: dict[str, int] = {}
    exclusion_counts: dict[str, int] = {}

    for subset in instruction_subsets:
        if subset not in INSTRUCTION_SUBSETS:
            raise ValueError(f"Unsupported GQA instruction subset: {subset}")
        schema = _instruction_schema(raw_root, subset=subset)
        split_specs = (
            (f"official_train_{subset}", False),
            (f"official_train_{subset}_without_mmeb_v2_eval", True),
        )
        for split_name, exclude_eval in split_specs:
            output_path = output_root / "data" / f"{split_name}.lance"
            if output_path.exists():
                if not overwrite:
                    raise FileExistsError(f"Output already exists: {output_path}")
                shutil.rmtree(output_path)
            split_counts[split_name] = _write_split(
                raw_root=raw_root,
                output_path=output_path,
                instruction_subset=subset,
                schema=schema,
                image_ids=image_ids,
                eval_overlap_keys=eval_overlap_keys,
                exclude_eval=exclude_eval,
                max_bytes_per_file=max_bytes_per_file,
            )

        exclusion_path = output_root / "data" / "exclusions" / f"mmeb_v2_eval_{subset}.lance"
        if exclusion_path.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {exclusion_path}")
            shutil.rmtree(exclusion_path)
        exclusion_counts[subset] = _write_eval_exclusions(
            raw_root=raw_root,
            output_path=exclusion_path,
            instruction_subset=subset,
            schema=schema,
            image_ids=image_ids,
            eval_overlap_keys=eval_overlap_keys,
            max_bytes_per_file=max_bytes_per_file,
        )

    return {
        "splits": split_counts,
        "exclusions": exclusion_counts,
        "available_image_ids": len(image_ids),
        "eval_overlap_keys": len(eval_overlap_keys),
    }


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_gqa_train_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the GQA runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing GQA image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_gqa_train_root(
    *,
    raw_root: Path,
    output_root: Path,
    eval_root: Path | None = None,
    overwrite: bool = False,
    image_batch_size: int = 1024,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert GQA into raw-preserving Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    split_summary = write_gqa_instruction_splits(
        raw_root=raw_root,
        output_root=output_root,
        eval_root=eval_root,
        overwrite=False,
        instruction_subsets=INSTRUCTION_SUBSETS,
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
    ensure_gqa_train_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "gqa_train_raw_lance_v1",
        "splits": split_summary["splits"],
        "exclusions": split_summary["exclusions"],
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
        "available_image_ids": split_summary["available_image_ids"],
        "eval_overlap_keys": split_summary["eval_overlap_keys"],
    }


__all__ = [
    "IMAGE_SCHEMA",
    "SUPPORTED_SPLITS",
    "ensure_gqa_train_indices",
    "load_gqa_eval_overlap_keys",
    "write_gqa_instruction_splits",
    "write_gqa_train_root",
]
