"""Wiki-SS-NQ training conversion utilities."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
from datasets import Image, load_from_disk

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

EVAL_QUERY_PREFIX = "Find the document image that can answer the given query: "
SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("query", pa.string()),
        pa.field("answers", pa.list_(pa.string())),
        pa.field("positive_rank", pa.int32()),
        pa.field("positive_docid", pa.string()),
        pa.field("positive_title", pa.string()),
        pa.field("positive_text", pa.string()),
        pa.field("negative_passages_json", pa.string()),
    ]
)
IMAGE_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("image", pa.binary()),
    ]
)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows as dictionaries."""

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def load_eval_overlap_queries(eval_root: Path | None) -> set[str]:
    """Load normalized MMEB-V2 eval queries when an eval Lance root is available."""

    if eval_root is None or not eval_root.exists():
        return set()
    queries_path = eval_root / "queries.lance"
    if not queries_path.exists():
        return set()
    dataset = lance.dataset(str(queries_path))
    text_column = "qry_text" if "qry_text" in dataset.schema.names else "text"
    if text_column not in dataset.schema.names:
        raise ValueError(f"Wiki-SS-NQ eval queries table has no text column: {queries_path}")
    normalized: set[str] = set()
    for row in dataset.to_table(columns=[text_column]).to_pylist():
        query = str(row.get(text_column, "") or "")
        if query.startswith(EVAL_QUERY_PREFIX):
            query = query[len(EVAL_QUERY_PREFIX) :]
        normalized.add(query.strip())
    return normalized


def _iter_sample_rows(query_root: Path) -> Iterator[dict[str, Any]]:
    """Yield one row per query-positive passage pair."""

    train_path = query_root / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing Wiki-SS-NQ train queries: {train_path}")
    for train_row in _iter_jsonl(train_path):
        query = str(train_row["query"]).strip()
        answers = [str(answer) for answer in train_row.get("answers", [])]
        negative_passages_json = json.dumps(train_row.get("negative_passages", []), ensure_ascii=False)
        for positive_rank, positive in enumerate(train_row.get("positive_passages", [])):
            yield {
                "query_id": str(train_row["query_id"]),
                "query": query,
                "answers": answers,
                "positive_rank": positive_rank,
                "positive_docid": str(positive["docid"]),
                "positive_title": str(positive.get("title", "") or ""),
                "positive_text": str(positive.get("text", "") or ""),
                "negative_passages_json": negative_passages_json,
            }


def _sample_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield sample rows as Arrow record batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SAMPLE_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SAMPLE_SCHEMA)


def _load_corpus(corpus_root: Path):
    """Load the screenshot corpus without decoding images through PIL."""

    dataset_dict = load_from_disk(str(corpus_root))
    return dataset_dict["train"].cast_column("image", Image(decode=False))


def _image_lookup_path(row: dict[str, Any]) -> str:
    """Return the lookup key used by training positives.

    The Wiki-SS-NQ corpus stores a sequential corpus row id in ``docid``, while
    the train query positives point to the Wikipedia screenshot id represented
    by the image filename stem. Keeping this key aligned at conversion time
    avoids runtime row skipping and keeps missing positives visible as data
    conversion failures.
    """

    image = row["image"]
    image_path = image.get("path") if isinstance(image, dict) else None
    if image_path:
        return Path(str(image_path)).stem
    return str(row["docid"])


def _image_batches(corpus_root: Path, *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield screenshot corpus rows as path/image records."""

    corpus = _load_corpus(corpus_root)
    rows: list[dict[str, Any]] = []
    for row in corpus:
        path = _image_lookup_path(row)
        image = row["image"]
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise TypeError(f"Wiki-SS-NQ corpus image {path!r} is missing image bytes.")
        rows.append({"path": path, "image": bytes(image_bytes)})
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


def ensure_wikissnq_train_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Wiki-SS-NQ runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing Wiki-SS-NQ image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_wikissnq_train_root(
    *,
    query_root: Path,
    corpus_root: Path,
    output_root: Path,
    eval_root: Path | None = None,
    overwrite: bool = False,
    sample_batch_size: int = 4096,
    image_batch_size: int = 1024,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert Wiki-SS-NQ into raw-preserving Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    eval_queries = load_eval_overlap_queries(eval_root)
    all_rows = list(_iter_sample_rows(query_root))
    train_rows = [row for row in all_rows if row["query"] not in eval_queries]
    excluded_rows = [row for row in all_rows if row["query"] in eval_queries]

    split_rows = {
        "official_train_raw": all_rows,
        "official_train": train_rows,
        "official_train_without_mmeb_v2_eval": train_rows,
    }
    for split_name, rows in split_rows.items():
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                SAMPLE_SCHEMA,
                _sample_batches(rows, batch_size=sample_batch_size),
            ),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SAMPLE_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            SAMPLE_SCHEMA,
            _sample_batches(excluded_rows, batch_size=sample_batch_size),
        ),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SAMPLE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            IMAGE_SCHEMA,
            _image_batches(corpus_root, batch_size=image_batch_size),
        ),
        str(output_root / "data" / "images.lance"),
        schema=IMAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_wikissnq_train_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "wikissnq_train_raw_lance_v1",
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "exclusion_rows": len(excluded_rows),
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
    }


__all__ = [
    "EVAL_QUERY_PREFIX",
    "IMAGE_SCHEMA",
    "SAMPLE_SCHEMA",
    "ensure_wikissnq_train_indices",
    "load_eval_overlap_queries",
    "write_wikissnq_train_root",
]
