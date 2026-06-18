"""RefCOCO training conversion utilities."""

from __future__ import annotations

import re
import shutil
import tarfile
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

SUPPORTED_SPLITS = (
    "official_train",
    "official_validation",
    "official_testA",
    "official_testB",
    "official_train_without_mmeb_v2_eval",
)
BBOX_TYPE = pa.list_(pa.float64())
INT_LIST_TYPE = pa.list_(pa.int64())
CAPTION_LIST_TYPE = pa.list_(pa.string())
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("ann_id", pa.int64()),
        pa.field("ref_id", pa.int64()),
        pa.field("image_id", pa.int64()),
        pa.field("split", pa.string()),
        pa.field("sent_ids", INT_LIST_TYPE),
        pa.field("captions", CAPTION_LIST_TYPE),
        pa.field("bbox", BBOX_TYPE),
        pa.field("category_id", pa.int64()),
        pa.field("file_name", pa.string()),
        pa.field("image_path", pa.string()),
        pa.field("global_image_id", pa.string()),
        pa.field("anns_id", pa.string()),
        pa.field("raw_anns", pa.string()),
        pa.field("raw_image_info", pa.string()),
        pa.field("raw_sentences", pa.string()),
    ]
)
IMAGES_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.int64()),
        pa.field("image", pa.binary()),
        pa.field("image_path", pa.string()),
    ]
)
_ANN_ID_RE = re.compile(r"image_(\d+)(?:_crop)?\.jpg$")


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


def _split_path(parquet_root: Path, split: str) -> Path:
    """Resolve one jxu124/refcoco parquet file by logical split."""

    patterns = {
        "official_train": "data/train-*.parquet",
        "official_validation": "data/validation-*.parquet",
        "official_testA": "data/test-*.parquet",
        "official_testB": "data/testB-*.parquet",
    }
    matches = sorted(parquet_root.glob(patterns[split]))
    if not matches:
        raise FileNotFoundError(f"Missing RefCOCO parquet for split {split}: {parquet_root}")
    if len(matches) != 1:
        raise ValueError(f"Expected one RefCOCO parquet for split {split}, found {len(matches)}.")
    return matches[0]


def _normalize_split_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one HF parquet row into the storage schema."""

    return {
        "ann_id": int(row["ann_id"]),
        "ref_id": int(row["ref_id"]),
        "image_id": int(row["image_id"]),
        "split": str(row["split"]),
        "sent_ids": [int(value) for value in row.get("sent_ids") or []],
        "captions": [str(value) for value in row.get("captions") or []],
        "bbox": [float(value) for value in row.get("bbox") or []],
        "category_id": int(row["category_id"]) if row.get("category_id") is not None else None,
        "file_name": str(row.get("file_name") or ""),
        "image_path": str(row.get("image_path") or ""),
        "global_image_id": str(row.get("global_image_id") or ""),
        "anns_id": str(row.get("anns_id") or ""),
        "raw_anns": str(row.get("raw_anns") or ""),
        "raw_image_info": str(row.get("raw_image_info") or ""),
        "raw_sentences": str(row.get("raw_sentences") or ""),
    }


def load_refcoco_split_rows(
    parquet_root: Path,
    *,
    excluded_ann_ids: set[int] | None = None,
) -> OrderedDict[str, list[dict[str, Any]]]:
    """Load RefCOCO split rows from the HF parquet mirror."""

    rows_by_split: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for split in ("official_train", "official_validation", "official_testA", "official_testB"):
        table = pq.read_table(_split_path(parquet_root, split))
        rows_by_split[split] = [_normalize_split_row(row) for row in table.to_pylist()]
    excluded = excluded_ann_ids or set()
    rows_by_split["official_train_without_mmeb_v2_eval"] = [
        row for row in rows_by_split["official_train"] if int(row["ann_id"]) not in excluded
    ]
    return rows_by_split


def collect_refcoco_eval_ann_ids(*eval_roots: Path) -> set[int]:
    """Collect annotation ids visible in MMEB-V2 RefCOCO eval artifacts."""

    ann_ids: set[int] = set()
    for root in eval_roots:
        for table_name, column in (("queries.lance", "qry_img_path"), ("candidates.lance", "tgt_img_path")):
            table_path = root / table_name
            if not table_path.exists():
                continue
            dataset = lance.dataset(str(table_path))
            if column not in dataset.schema.names:
                continue
            rows = dataset.to_table(columns=[column]).to_pylist()
            for row in rows:
                value = row.get(column)
                if not value:
                    continue
                match = _ANN_ID_RE.search(str(value))
                if match:
                    ann_ids.add(int(match.group(1)))
    return ann_ids


def collect_image_ids(rows_by_split: OrderedDict[str, list[dict[str, Any]]]) -> set[int]:
    """Collect unique COCO image ids referenced by split rows."""

    return set(collect_image_refs(rows_by_split))


def collect_image_refs(rows_by_split: OrderedDict[str, list[dict[str, Any]]]) -> OrderedDict[int, str]:
    """Collect unique COCO image ids and official train2014 member paths."""

    refs: OrderedDict[int, str] = OrderedDict()
    image_ids: set[int] = set()
    for rows in rows_by_split.values():
        for row in rows:
            image_id = int(row["image_id"])
            if image_id in image_ids:
                continue
            image_ids.add(image_id)
            image_path = str(row.get("image_path") or "")
            if image_path.startswith("coco/"):
                image_path = image_path[len("coco/") :]
            refs[image_id] = image_path or f"train2014/COCO_train2014_{image_id:012d}.jpg"
    return refs


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SPLIT_SCHEMA)


def _iter_image_rows_from_zip(*, zip_path: Path, image_refs: Mapping[int, str]) -> Iterator[dict[str, Any]]:
    """Yield selected COCO image rows from the official train2014 zip."""

    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        missing: list[str] = []
        for image_id, member_path in image_refs.items():
            candidates = [member_path]
            if member_path.startswith("train2014/"):
                candidates.append(member_path.split("/", 1)[1])
            selected = next((candidate for candidate in candidates if candidate in members), None)
            if selected is None:
                missing.append(member_path)
                continue
            yield {
                "image_id": int(image_id),
                "image": archive.read(selected),
                "image_path": selected,
            }
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing COCO image members in train2014 zip: {preview}")


def _tar_member_candidates(member_path: str) -> list[str]:
    """Return possible RefCOCO HF tar member paths for one COCO image."""

    normalized = member_path.lstrip("./")
    candidates = [normalized]
    if normalized.endswith(".jpg"):
        candidates.append(normalized[:-4])
    if normalized.startswith("train2014/"):
        candidates.append(normalized.split("/", 1)[1])
    candidates.extend(f"refcoco/images/{candidate}" for candidate in list(candidates))
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _iter_image_rows_from_tar(*, tar_path: Path, image_refs: Mapping[int, str]) -> Iterator[dict[str, Any]]:
    """Yield selected COCO image rows from a RefCOCO HF image tar."""

    image_id_by_member: dict[str, int] = {}
    for image_id, member_path in image_refs.items():
        for candidate in _tar_member_candidates(member_path):
            image_id_by_member[candidate] = int(image_id)

    found: set[int] = set()
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = member.name.lstrip("./")
            image_id = image_id_by_member.get(member_name)
            if image_id is None:
                continue
            payload = archive.extractfile(member)
            if payload is None:
                continue
            found.add(image_id)
            yield {
                "image_id": image_id,
                "image": payload.read(),
                "image_path": member_name,
            }
    missing = sorted(set(image_refs) - found)
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise FileNotFoundError(f"Missing RefCOCO image ids in image tar: {preview}")


def _iter_image_rows_from_parquet(*, coco_images_root: Path, image_refs: Mapping[int, str]) -> Iterator[dict[str, Any]]:
    """Yield selected COCO image rows from HF parquet shards when ids match original COCO ids."""

    found: set[int] = set()
    image_ids = set(image_refs)
    files = sorted((coco_images_root / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing COCO image parquet shards: {coco_images_root}")
    for parquet_path in files:
        table = pq.read_table(parquet_path, columns=["image_id", "image"])
        for row in table.to_pylist():
            image_id = int(row["image_id"])
            if image_id not in image_ids:
                continue
            image = row.get("image") or {}
            image_bytes = image.get("bytes")
            if not isinstance(image_bytes, (bytes, bytearray)):
                raise TypeError(f"COCO image row is missing bytes: image_id={image_id}")
            found.add(image_id)
            yield {
                "image_id": image_id,
                "image": bytes(image_bytes),
                "image_path": str(image.get("path") or ""),
            }
    missing = sorted(image_ids - found)
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise FileNotFoundError(f"Missing COCO image ids in image parquet shards: {preview}")


def _iter_image_rows(*, coco_images_root: Path, image_refs: Mapping[int, str]) -> Iterator[dict[str, Any]]:
    """Yield selected COCO image rows from an official zip or compatible parquet mirror."""

    zip_path = coco_images_root / "train2014.zip"
    if zip_path.is_file():
        yield from _iter_image_rows_from_zip(zip_path=zip_path, image_refs=image_refs)
        return
    for tar_name in ("refcoco.tar.gz", "refcoco.tar", "train2014.tar", "train2014.tar.gz"):
        tar_path = coco_images_root / tar_name
        if tar_path.is_file():
            yield from _iter_image_rows_from_tar(tar_path=tar_path, image_refs=image_refs)
            return
    yield from _iter_image_rows_from_parquet(coco_images_root=coco_images_root, image_refs=image_refs)


def _image_batches(
    *,
    coco_images_root: Path,
    image_refs: Mapping[int, str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield image table batches."""

    for batch in _batched(_iter_image_rows(coco_images_root=coco_images_root, image_refs=image_refs), batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=IMAGES_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_refcoco_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the RefCOCO runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing RefCOCO image table: {images_path}")
    _create_scalar_index(images_path, "image_id")


def write_refcoco_root(
    *,
    parquet_root: Path,
    coco_images_root: Path,
    output_root: Path,
    mmeb_v2_eval_roots: list[Path] | None = None,
    overwrite: bool = False,
    image_batch_size: int = 1024,
    split_batch_size: int = 4096,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert RefCOCO into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    excluded_ann_ids = collect_refcoco_eval_ann_ids(*(mmeb_v2_eval_roots or []))
    rows_by_split = load_refcoco_split_rows(parquet_root, excluded_ann_ids=excluded_ann_ids)
    image_refs = collect_image_refs(rows_by_split)

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            IMAGES_SCHEMA,
            _image_batches(
                coco_images_root=coco_images_root,
                image_refs=image_refs,
                batch_size=image_batch_size,
            ),
        ),
        str(output_root / "data" / "images.lance"),
        schema=IMAGES_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    for split_name, rows in rows_by_split.items():
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(rows, batch_size=split_batch_size)),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    exclusion_rows = [row for row in rows_by_split["official_train"] if int(row["ann_id"]) in excluded_ann_ids]
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(exclusion_rows, batch_size=split_batch_size)),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_refcoco_indices(output_root)
    return {
        "output_root": str(output_root),
        "image_rows": len(image_refs),
        "splits": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "exclusion_rows": len(exclusion_rows),
        "artifact_format": "refcoco_train_raw_lance_v1",
    }


__all__ = [
    "IMAGES_SCHEMA",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "collect_image_ids",
    "collect_image_refs",
    "collect_refcoco_eval_ann_ids",
    "ensure_refcoco_indices",
    "load_refcoco_split_rows",
    "write_refcoco_root",
]
