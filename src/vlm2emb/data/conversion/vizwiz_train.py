"""VizWiz VQA training conversion utilities."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

IMAGE_SCHEMA = pa.schema([pa.field("path", pa.string()), pa.field("image", pa.binary())])
STRING_LIST_TYPE = pa.list_(pa.string())
SUPPORTED_SPLITS = (
    "official_train",
    "official_validation",
    "official_test",
    "official_train_answerable",
    "official_validation_answerable",
    "official_train_answerable_without_mmeb_v2_eval",
)
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("image", pa.string()),
        pa.field("image_split", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answers", STRING_LIST_TYPE),
        pa.field("answer_confidences", STRING_LIST_TYPE),
        pa.field("answer_type", pa.string()),
        pa.field("answerable", pa.bool_()),
        pa.field("selected_answer", pa.string()),
        pa.field("raw_answers", pa.string()),
    ]
)
DEFAULT_QUERY_PREFIX = "Represent the given image with the following question:"
CONFIDENCE_WEIGHTS = {"yes": 3, "maybe": 1, "no": 0}


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
    """Normalize text enough for overlap matching and answer voting."""

    return " ".join(text.strip().split()).lower()


def _normalize_eval_question(text: str) -> str:
    """Recover the raw VizWiz question from a MMEB-V2 query string."""

    normalized = text.replace("<|image_1|>", "").replace("<|image_pad|>", "").strip()
    if normalized.startswith(DEFAULT_QUERY_PREFIX):
        normalized = normalized[len(DEFAULT_QUERY_PREFIX) :].strip()
    return _normalize_text(normalized)


def _selected_answer(answers: list[dict[str, Any]], *, answerable: bool) -> str:
    """Select one answer with confidence-aware majority voting."""

    if not answerable:
        return ""
    scores: Counter[str] = Counter()
    first_seen: dict[str, str] = {}
    for answer in answers:
        value = str(answer.get("answer", "") or "").strip()
        normalized = _normalize_text(value)
        if not normalized or normalized == "unanswerable":
            continue
        confidence = str(answer.get("answer_confidence", "") or "").strip().lower()
        weight = CONFIDENCE_WEIGHTS.get(confidence, 0)
        if weight <= 0:
            continue
        scores[normalized] += weight
        first_seen.setdefault(normalized, value)
    if not scores:
        return ""
    selected, _ = max(scores.items(), key=lambda item: (item[1], -list(first_seen).index(item[0])))
    return first_seen[selected]


def _annotation_zip_path(raw_root: Path) -> Path:
    """Resolve the VizWiz annotation archive."""

    path = raw_root / "Annotations.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing VizWiz annotation archive: {path}")
    return path


def _annotation_rows(raw_root: Path, split: str) -> list[dict[str, Any]]:
    """Load one official VizWiz annotation JSON from the archive."""

    annotation_name = {"train": "train.json", "validation": "val.json", "test": "test.json"}[split]
    with zipfile.ZipFile(_annotation_zip_path(raw_root)) as archive:
        rows = json.loads(archive.read(annotation_name))
    return [_normalize_annotation_row(row, split=split) for row in rows]


def _normalize_annotation_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Normalize one official VizWiz annotation row."""

    raw_answers = list(row.get("answers") or [])
    answerable = bool(row.get("answerable", False))
    return {
        "image": str(row.get("image") or ""),
        "image_split": split,
        "question": str(row.get("question") or ""),
        "answers": [str(answer.get("answer", "") or "") for answer in raw_answers],
        "answer_confidences": [str(answer.get("answer_confidence", "") or "") for answer in raw_answers],
        "answer_type": str(row.get("answer_type") or ""),
        "answerable": answerable,
        "selected_answer": _selected_answer(raw_answers, answerable=answerable),
        "raw_answers": json.dumps(raw_answers, ensure_ascii=False, sort_keys=True),
    }


def load_vizwiz_eval_overlap_keys(eval_root: Path | None) -> set[tuple[str, str]]:
    """Load MMEB-V2 eval `(question, answer)` keys for train exclusion."""

    if eval_root is None:
        return set()
    queries_path = eval_root / "queries.lance"
    candidates_path = eval_root / "candidates.lance"
    qrels_path = eval_root / "qrels.lance"
    for path in (queries_path, candidates_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing VizWiz MMEB-V2 table: {path}")

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


def _row_overlaps_eval(row: dict[str, Any], eval_overlap_keys: set[tuple[str, str]]) -> bool:
    """Return whether one VizWiz row overlaps MMEB-V2 eval."""

    if not eval_overlap_keys:
        return False
    key = (_normalize_text(str(row.get("question", "") or "")), _normalize_text(str(row.get("selected_answer", "") or "")))
    return key in eval_overlap_keys


def load_vizwiz_split_rows(
    raw_root: Path,
    *,
    eval_overlap_keys: set[tuple[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load official and training-filtered VizWiz split rows."""

    rows_by_split = {
        "official_train": _annotation_rows(raw_root, "train"),
        "official_validation": _annotation_rows(raw_root, "validation"),
        "official_test": _annotation_rows(raw_root, "test"),
    }
    rows_by_split["official_train_answerable"] = [
        row for row in rows_by_split["official_train"] if row["answerable"] and row["selected_answer"]
    ]
    rows_by_split["official_validation_answerable"] = [
        row for row in rows_by_split["official_validation"] if row["answerable"] and row["selected_answer"]
    ]
    overlaps = eval_overlap_keys or set()
    rows_by_split["official_train_answerable_without_mmeb_v2_eval"] = [
        row for row in rows_by_split["official_train_answerable"] if not _row_overlaps_eval(row, overlaps)
    ]
    return rows_by_split


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SPLIT_SCHEMA)


def _image_zip_path(raw_root: Path, split: str) -> Path:
    """Resolve the official VizWiz image archive for one split."""

    return raw_root / f"{'val' if split == 'validation' else split}.zip"


def _iter_image_rows(raw_root: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """Yield unique image bytes referenced by train and validation annotations."""

    seen: set[str] = set()
    for split, split_name in (("train", "official_train"), ("validation", "official_validation")):
        zip_path = _image_zip_path(raw_root, split)
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing VizWiz image archive: {zip_path}")
        needed = {str(row["image"]) for row in rows_by_split[split_name]}
        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
            for image_name in sorted(needed):
                if image_name in seen:
                    continue
                candidates = [f"{'val' if split == 'validation' else split}/{image_name}", image_name]
                selected = next((candidate for candidate in candidates if candidate in members), None)
                if selected is None:
                    raise FileNotFoundError(f"Missing VizWiz image member in {zip_path}: {image_name}")
                seen.add(image_name)
                yield {"path": image_name, "image": archive.read(selected)}


def _image_batches(
    raw_root: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield VizWiz image table batches."""

    for rows in _batched(_iter_image_rows(raw_root, rows_by_split), batch_size):
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_vizwiz_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the VizWiz runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing VizWiz image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_vizwiz_root(
    *,
    raw_root: Path,
    output_root: Path,
    eval_root: Path | None = None,
    overwrite: bool = False,
    split_batch_size: int = 4096,
    image_batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert VizWiz VQA into raw-preserving Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    eval_overlap_keys = load_vizwiz_eval_overlap_keys(eval_root)
    rows_by_split = load_vizwiz_split_rows(raw_root, eval_overlap_keys=eval_overlap_keys)
    for split_name, rows in rows_by_split.items():
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(rows, batch_size=split_batch_size)),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    exclusion_rows = [row for row in rows_by_split["official_train_answerable"] if _row_overlaps_eval(row, eval_overlap_keys)]
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(exclusion_rows, batch_size=split_batch_size)),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            IMAGE_SCHEMA,
            _image_batches(raw_root, rows_by_split, batch_size=image_batch_size),
        ),
        str(output_root / "data" / "images.lance"),
        schema=IMAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_vizwiz_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "vizwiz_train_raw_lance_v1",
        "splits": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "exclusion_rows": len(exclusion_rows),
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
        "eval_overlap_keys": len(eval_overlap_keys),
    }


__all__ = [
    "IMAGE_SCHEMA",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "ensure_vizwiz_indices",
    "load_vizwiz_eval_overlap_keys",
    "load_vizwiz_split_rows",
    "write_vizwiz_root",
]
