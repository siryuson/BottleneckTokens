"""TextVQA training conversion utilities."""

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
    "official_train",
    "official_validation",
    "official_test",
    "official_train_without_mmeb_v2_eval",
)
DEFAULT_QUERY_PREFIX = "Represent the given image with the following question:"


def _split_files(raw_root: Path, split: str) -> list[Path]:
    """Return sorted TextVQA parquet files for one split."""

    files = sorted(raw_root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing TextVQA split files: {raw_root}/{split}-*.parquet")
    return files


def _split_schema(raw_root: Path) -> pa.Schema:
    """Return the raw schema with embedded image bytes moved to the image table."""

    schema = pq.ParquetFile(_split_files(raw_root, "train")[0]).schema_arrow
    return schema.remove(schema.get_field_index("image"))


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


def _normalize_text(text: str) -> str:
    """Normalize text enough for eval-overlap matching."""

    return " ".join(text.strip().split()).lower()


def _normalize_eval_question(text: str) -> str:
    """Recover the raw TextVQA question from a MMEB-V2 query string."""

    normalized = text.replace("<|image_1|>", "").replace("<|image_pad|>", "").strip()
    if normalized.startswith(DEFAULT_QUERY_PREFIX):
        normalized = normalized[len(DEFAULT_QUERY_PREFIX) :].strip()
    return _normalize_text(normalized)


def load_textvqa_eval_overlap_keys(eval_root: Path | None) -> set[tuple[str, str]]:
    """Load MMEB-V2 eval `(question, answer)` keys for train exclusion."""

    if eval_root is None:
        return set()
    queries_path = eval_root / "queries.lance"
    candidates_path = eval_root / "candidates.lance"
    qrels_path = eval_root / "qrels.lance"
    for path in (queries_path, candidates_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing TextVQA MMEB-V2 table: {path}")

    query_rows = lance.dataset(str(queries_path)).to_table(columns=["id", "qry_text"]).to_pylist()
    candidate_rows = lance.dataset(str(candidates_path)).to_table(columns=["id", "text"]).to_pylist()
    qrels_rows = lance.dataset(str(qrels_path)).to_table(columns=["query_id", "candidate_ids", "candidate_scores"]).to_pylist()
    question_by_id = {int(row["id"]): _normalize_eval_question(str(row.get("qry_text", "") or "")) for row in query_rows}
    answer_by_id = {int(row["id"]): _normalize_text(str(row.get("text", "") or "")) for row in candidate_rows}

    overlap_keys: set[tuple[str, str]] = set()
    for row in qrels_rows:
        question = question_by_id.get(int(row["query_id"]))
        if not question:
            continue
        for candidate_id, score in zip(row.get("candidate_ids") or [], row.get("candidate_scores") or [], strict=True):
            if float(score) <= 0:
                continue
            answer = answer_by_id.get(int(candidate_id))
            if answer:
                overlap_keys.add((question, answer))
    return overlap_keys


def _row_overlaps_eval(question: str, answers: list[str], eval_overlap_keys: set[tuple[str, str]]) -> bool:
    """Return whether a TextVQA row overlaps MMEB-V2 eval."""

    if not eval_overlap_keys:
        return False
    normalized_question = _normalize_text(question)
    return any((normalized_question, _normalize_text(answer)) in eval_overlap_keys for answer in answers)


def _strip_image_column(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Remove embedded image bytes from one raw batch."""

    return pa.Table.from_batches([batch]).drop(["image"]).cast(schema).to_batches()[0]


def _split_batches(
    raw_root: Path,
    split: str,
    *,
    schema: pa.Schema,
    eval_overlap_keys: set[tuple[str, str]],
    exclude_eval: bool,
) -> Iterator[pa.RecordBatch]:
    """Yield TextVQA split batches with image bytes moved to the side table."""

    emitted = False
    for path in _split_files(raw_root, split):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            if exclude_eval:
                questions = [str(value) for value in batch.column("question").to_pylist()]
                answers = [[str(answer) for answer in (value or [])] for value in batch.column("answers").to_pylist()]
                mask = [
                    not _row_overlaps_eval(question, answer_list, eval_overlap_keys)
                    for question, answer_list in zip(questions, answers, strict=True)
                ]
                batch = batch.filter(pa.array(mask, type=pa.bool_()))
            if batch.num_rows:
                emitted = True
                yield _strip_image_column(batch, schema)
    if not emitted:
        yield pa.RecordBatch.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


def _exclusion_batches(
    raw_root: Path,
    *,
    schema: pa.Schema,
    eval_overlap_keys: set[tuple[str, str]],
) -> Iterator[pa.RecordBatch]:
    """Yield TextVQA rows removed from train because they overlap MMEB-V2 eval."""

    emitted = False
    for path in _split_files(raw_root, "train"):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            questions = [str(value) for value in batch.column("question").to_pylist()]
            answers = [[str(answer) for answer in (value or [])] for value in batch.column("answers").to_pylist()]
            mask = [
                _row_overlaps_eval(question, answer_list, eval_overlap_keys)
                for question, answer_list in zip(questions, answers, strict=True)
            ]
            filtered = batch.filter(pa.array(mask, type=pa.bool_()))
            if filtered.num_rows:
                emitted = True
                yield _strip_image_column(filtered, schema)
    if not emitted:
        yield pa.RecordBatch.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


def _iter_image_rows(raw_root: Path) -> Iterator[dict[str, Any]]:
    """Yield unique TextVQA image table rows using image_id as the lookup path."""

    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        for path in _split_files(raw_root, split):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(columns=["image_id", "image"]):
                for row in pa.Table.from_batches([batch]).to_pylist():
                    image_id = str(row.get("image_id", "") or "")
                    if not image_id or image_id in seen:
                        continue
                    image = row.get("image") or {}
                    payload = image.get("bytes") if isinstance(image, dict) else None
                    if payload is None:
                        continue
                    seen.add(image_id)
                    yield {"path": image_id, "image": bytes(payload)}


def _image_batches(raw_root: Path, *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield TextVQA image table rows as Arrow batches."""

    for rows in _batched(_iter_image_rows(raw_root), batch_size):
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_textvqa_train_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the TextVQA runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing TextVQA image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_textvqa_train_root(
    *,
    raw_root: Path,
    output_root: Path,
    eval_root: Path | None = None,
    overwrite: bool = False,
    split_batch_size: int = 4096,
    image_batch_size: int = 1024,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert TextVQA into raw-preserving Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    schema = _split_schema(raw_root)
    eval_overlap_keys = load_textvqa_eval_overlap_keys(eval_root)
    split_map = {
        "official_train": ("train", False),
        "official_validation": ("validation", False),
        "official_test": ("test", False),
        "official_train_without_mmeb_v2_eval": ("train", True),
    }
    split_counts: dict[str, int] = {}
    for output_split, (raw_split, exclude_eval) in split_map.items():
        output_path = output_root / "data" / f"{output_split}.lance"
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                schema,
                _split_batches(
                    raw_root,
                    raw_split,
                    schema=schema,
                    eval_overlap_keys=eval_overlap_keys,
                    exclude_eval=exclude_eval,
                ),
            ),
            str(output_path),
            schema=schema,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
        split_counts[output_split] = lance.dataset(str(output_path)).count_rows()

    exclusion_path = output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            schema,
            _exclusion_batches(raw_root, schema=schema, eval_overlap_keys=eval_overlap_keys),
        ),
        str(exclusion_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(IMAGE_SCHEMA, _image_batches(raw_root, batch_size=image_batch_size)),
        str(output_root / "data" / "images.lance"),
        schema=IMAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_textvqa_train_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "textvqa_train_raw_lance_v1",
        "splits": split_counts,
        "exclusion_rows": lance.dataset(str(exclusion_path)).count_rows(),
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
        "eval_overlap_keys": len(eval_overlap_keys),
    }


__all__ = [
    "IMAGE_SCHEMA",
    "SUPPORTED_SPLITS",
    "ensure_textvqa_train_indices",
    "load_textvqa_eval_overlap_keys",
    "write_textvqa_train_root",
]
