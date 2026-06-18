"""M-BEIR training conversion utilities."""

from __future__ import annotations

import json
import shutil
import tarfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

QUERY_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.string()),
        pa.field("query_txt", pa.string()),
        pa.field("query_img_path", pa.string()),
        pa.field("query_modality", pa.string()),
        pa.field("query_src_content", pa.string()),
        pa.field("pos_cand_list", pa.list_(pa.string())),
        pa.field("neg_cand_list", pa.list_(pa.string())),
        pa.field("task_id", pa.int32()),
    ]
)
CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("did", pa.string()),
        pa.field("txt", pa.string()),
        pa.field("img_path", pa.string()),
        pa.field("modality", pa.string()),
        pa.field("src_content", pa.string()),
    ]
)
IMAGE_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("image", pa.binary()),
    ]
)


@dataclass(frozen=True)
class MbeirTaskConfig:
    """Source mapping for one archive-aligned M-BEIR training task."""

    dataset_name: str
    task_id: int
    query_file: str
    candidate_file: str
    test_query_file: str | None = None


MBEIR_TASK_CONFIGS: OrderedDict[str, MbeirTaskConfig] = OrderedDict(
    [
        (
            "FashionIQ_train",
            MbeirTaskConfig(
                dataset_name="FashionIQ_train",
                task_id=7,
                query_file="query/train/mbeir_fashioniq_train.jsonl",
                candidate_file="cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl",
                test_query_file="query/test/mbeir_fashioniq_task7_test.jsonl",
            ),
        ),
        (
            "Fashion200K_t2i",
            MbeirTaskConfig(
                dataset_name="Fashion200K_t2i",
                task_id=0,
                query_file="query/train/mbeir_fashion200k_train.jsonl",
                candidate_file="cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl",
                test_query_file="query/test/mbeir_fashion200k_task0_test.jsonl",
            ),
        ),
        (
            "Fashion200K_i2t",
            MbeirTaskConfig(
                dataset_name="Fashion200K_i2t",
                task_id=3,
                query_file="query/train/mbeir_fashion200k_train.jsonl",
                candidate_file="cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl",
                test_query_file="query/test/mbeir_fashion200k_task3_test.jsonl",
            ),
        ),
        (
            "OVEN_it2it_train",
            MbeirTaskConfig(
                dataset_name="OVEN_it2it_train",
                task_id=8,
                query_file="query/train/mbeir_oven_train.jsonl",
                candidate_file="cand_pool/local/mbeir_oven_task8_cand_pool.jsonl",
                test_query_file="query/test/mbeir_oven_task8_test.jsonl",
            ),
        ),
        (
            "OVEN_it2t_train",
            MbeirTaskConfig(
                dataset_name="OVEN_it2t_train",
                task_id=6,
                query_file="query/train/mbeir_oven_train.jsonl",
                candidate_file="cand_pool/local/mbeir_oven_task6_cand_pool.jsonl",
                test_query_file="query/test/mbeir_oven_task6_test.jsonl",
            ),
        ),
        (
            "EDIS_train",
            MbeirTaskConfig(
                dataset_name="EDIS_train",
                task_id=2,
                query_file="query/train/mbeir_edis_train.jsonl",
                candidate_file="cand_pool/local/mbeir_edis_task2_cand_pool.jsonl",
                test_query_file="query/test/mbeir_edis_task2_test.jsonl",
            ),
        ),
        (
            "INFOSEEK_it2t",
            MbeirTaskConfig(
                dataset_name="INFOSEEK_it2t",
                task_id=6,
                query_file="query/train/mbeir_infoseek_train.jsonl",
                candidate_file="cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl",
                test_query_file="query/test/mbeir_infoseek_task6_test.jsonl",
            ),
        ),
        (
            "INFOSEEK_it2it",
            MbeirTaskConfig(
                dataset_name="INFOSEEK_it2it",
                task_id=8,
                query_file="query/train/mbeir_infoseek_train.jsonl",
                candidate_file="cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl",
                test_query_file="query/test/mbeir_infoseek_task8_test.jsonl",
            ),
        ),
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


def _first_positive_id(row: dict[str, Any]) -> str:
    """Return the first positive candidate id used by the origin parser."""

    pos_cand_list = row.get("pos_cand_list") or []
    if not pos_cand_list:
        raise ValueError(f"M-BEIR query row has no positive candidate: {row.get('qid')!r}")
    return str(pos_cand_list[0])


def _normalize_query_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the query JSON fields used by the archive parser."""

    return {
        "qid": str(row.get("qid", "") or ""),
        "query_txt": row.get("query_txt"),
        "query_img_path": row.get("query_img_path"),
        "query_modality": str(row.get("query_modality", "") or ""),
        "query_src_content": row.get("query_src_content"),
        "pos_cand_list": [str(value) for value in (row.get("pos_cand_list") or [])],
        "neg_cand_list": [str(value) for value in (row.get("neg_cand_list") or [])],
        "task_id": int(row.get("task_id")),
    }


def _normalize_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the candidate JSON fields used by the archive parser."""

    return {
        "did": str(row.get("did", "") or ""),
        "txt": row.get("txt"),
        "img_path": row.get("img_path"),
        "modality": str(row.get("modality", "") or ""),
        "src_content": row.get("src_content"),
    }


def _load_eval_qids(raw_root: Path, config: MbeirTaskConfig) -> set[str]:
    """Load same-task test qids used as the eval exclusion boundary."""

    if not config.test_query_file:
        return set()
    path = raw_root / config.test_query_file
    if not path.exists():
        return set()
    qids: set[str] = set()
    for row in _iter_jsonl(path):
        if int(row.get("task_id", config.task_id)) == config.task_id:
            qids.add(str(row.get("qid", "") or ""))
    return qids


def _iter_task_query_rows(
    raw_root: Path,
    config: MbeirTaskConfig,
) -> Iterator[dict[str, Any]]:
    """Yield archive-aligned query rows for one M-BEIR task."""

    path = raw_root / config.query_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing M-BEIR query file: {path}")
    for row in _iter_jsonl(path):
        if int(row.get("task_id", -1)) != config.task_id:
            continue
        normalized = _normalize_query_row(row)
        _first_positive_id(normalized)
        yield normalized


def _query_batches(
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield query rows as Arrow record batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=QUERY_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=QUERY_SCHEMA)


def _candidate_batches(
    *,
    raw_root: Path,
    config: MbeirTaskConfig,
    required_candidate_ids: set[str],
    image_paths: set[str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield selected positive candidate rows and collect their image paths."""

    path = raw_root / config.candidate_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing M-BEIR candidate file: {path}")

    matched: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw_row in _iter_jsonl(path):
        did = str(raw_row.get("did", "") or "")
        if did not in required_candidate_ids:
            continue
        row = _normalize_candidate_row(raw_row)
        matched.add(did)
        img_path = row.get("img_path")
        if img_path:
            image_paths.add(str(img_path))
        rows.append(row)
        if len(rows) >= batch_size:
            yield pa.RecordBatch.from_pylist(rows, schema=CANDIDATE_SCHEMA)
            rows = []
    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=CANDIDATE_SCHEMA)

    missing = required_candidate_ids - matched
    if missing:
        preview = sorted(missing)[:10]
        raise KeyError(
            f"M-BEIR task {config.dataset_name} is missing {len(missing)} positive candidates; "
            f"first missing ids: {preview}"
        )


def _read_image_row(raw_root: Path, image_path: str) -> dict[str, Any]:
    """Read one image payload from the extracted M-BEIR image tree."""

    path = raw_root / image_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing M-BEIR image file: {path}")
    return {"path": image_path, "image": path.read_bytes()}


class _MultiPartReader:
    """Read a split archive as one continuous byte stream."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.index = 0
        self.handle = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while size < 0 or remaining > 0:
            if self.handle is None:
                if self.index >= len(self.paths):
                    break
                self.handle = self.paths[self.index].open("rb")
                self.index += 1
            data = self.handle.read(-1 if size < 0 else remaining)
            if data:
                chunks.append(data)
                if size >= 0:
                    remaining -= len(data)
                    if remaining <= 0:
                        break
                continue
            self.handle.close()
            self.handle = None
        return b"".join(chunks)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _archive_part_paths(raw_root: Path) -> list[Path]:
    """Return sorted split archive parts for the M-BEIR image tarball."""

    return sorted(raw_root.glob("mbeir_images.tar.gz.part-*"))


def _archive_image_batches(
    *,
    raw_root: Path,
    missing_image_paths: set[str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Stream missing images out of the split M-BEIR tar.gz archive."""

    archive_parts = _archive_part_paths(raw_root)
    if not archive_parts:
        preview = sorted(missing_image_paths)[:10]
        raise FileNotFoundError(
            f"M-BEIR image files are missing and no split archive parts were found; "
            f"first missing paths: {preview}"
        )

    remaining = set(missing_image_paths)
    rows: list[dict[str, Any]] = []
    reader = _MultiPartReader(archive_parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if member.isdir() or member.name not in remaining:
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    raise ValueError(f"Unable to read M-BEIR archive member: {member.name}")
                rows.append({"path": member.name, "image": file_obj.read()})
                remaining.remove(member.name)
                if len(rows) >= batch_size:
                    yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)
                    rows = []
                if not remaining:
                    break
    finally:
        reader.close()

    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGE_SCHEMA)
    if remaining:
        preview = sorted(remaining)[:10]
        raise FileNotFoundError(
            f"M-BEIR split archive did not contain {len(remaining)} required images; "
            f"first missing paths: {preview}"
        )


def _image_batches(
    *,
    raw_root: Path,
    image_paths: Iterable[str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Read selected M-BEIR images in parallel and yield Arrow batches."""

    sorted_paths = sorted(set(path for path in image_paths if path))
    file_paths = [image_path for image_path in sorted_paths if (raw_root / image_path).is_file()]
    missing_paths = set(sorted_paths) - set(file_paths)
    if num_workers <= 1:
        row_iter = (_read_image_row(raw_root, image_path) for image_path in file_paths)
        for batch in _batched(row_iter, batch_size):
            yield pa.RecordBatch.from_pylist(batch, schema=IMAGE_SCHEMA)
    elif file_paths:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            row_iter = pool.map(lambda image_path: _read_image_row(raw_root, image_path), file_paths)
            for batch in _batched(row_iter, batch_size):
                yield pa.RecordBatch.from_pylist(batch, schema=IMAGE_SCHEMA)
    if missing_paths:
        yield from _archive_image_batches(
            raw_root=raw_root,
            missing_image_paths=missing_paths,
            batch_size=batch_size,
        )


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_mbeir_train_indices(output_root: Path, dataset_names: Iterable[str] | None = None) -> None:
    """Ensure lookup indices required by the M-BEIR runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing M-BEIR image table: {images_path}")
    _create_scalar_index(images_path, "path")

    names = list(dataset_names or MBEIR_TASK_CONFIGS)
    for dataset_name in names:
        candidates_path = output_root / "data" / "candidates" / f"{dataset_name}.lance"
        if not candidates_path.exists():
            raise FileNotFoundError(f"Missing M-BEIR candidate table: {candidates_path}")
        _create_scalar_index(candidates_path, "did")


def write_mbeir_train_root(
    *,
    raw_root: Path,
    output_root: Path,
    dataset_names: list[str] | None = None,
    overwrite: bool = False,
    query_batch_size: int = 8192,
    candidate_batch_size: int = 8192,
    image_batch_size: int = 1024,
    num_workers: int = 8,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert selected M-BEIR train tasks into raw-preserving Lance tables."""

    selected_names = dataset_names or list(MBEIR_TASK_CONFIGS)
    unknown = sorted(set(selected_names) - set(MBEIR_TASK_CONFIGS))
    if unknown:
        raise ValueError(f"Unsupported M-BEIR dataset names: {unknown}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "candidates").mkdir(parents=True, exist_ok=True)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    image_paths: set[str] = set()
    split_counts: dict[str, dict[str, int]] = {}
    required_candidates_by_dataset: dict[str, set[str]] = {}

    for dataset_name in selected_names:
        config = MBEIR_TASK_CONFIGS[dataset_name]
        (output_root / "data" / dataset_name).mkdir(parents=True, exist_ok=True)
        eval_qids = _load_eval_qids(raw_root, config)
        all_rows = list(_iter_task_query_rows(raw_root, config))
        train_rows = [row for row in all_rows if row["qid"] not in eval_qids]
        for row in all_rows:
            if row.get("query_img_path"):
                image_paths.add(str(row["query_img_path"]))
        required_candidates_by_dataset[dataset_name] = {
            _first_positive_id(row)
            for row in all_rows
        }
        split_counts[dataset_name] = {
            "official_train": len(all_rows),
            "official_train_without_mmeb_v2_eval": len(train_rows),
            "mmeb_v2_eval_exclusion": len(all_rows) - len(train_rows),
        }

        for split_name, rows in (
            ("official_train", all_rows),
            ("official_train_without_mmeb_v2_eval", train_rows),
        ):
            lance.write_dataset(
                pa.RecordBatchReader.from_batches(
                    QUERY_SCHEMA,
                    _query_batches(rows, batch_size=query_batch_size),
                ),
                str(output_root / "data" / dataset_name / f"{split_name}.lance"),
                schema=QUERY_SCHEMA,
                mode="create",
                max_bytes_per_file=max_bytes_per_file,
            )

        excluded_rows = [row for row in all_rows if row["qid"] in eval_qids]
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                QUERY_SCHEMA,
                _query_batches(excluded_rows, batch_size=query_batch_size),
            ),
            str(output_root / "data" / "exclusions" / f"{dataset_name}_mmeb_v2_eval.lance"),
            schema=QUERY_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    candidate_counts: dict[str, int] = {}
    for dataset_name in selected_names:
        config = MBEIR_TASK_CONFIGS[dataset_name]
        batches = _candidate_batches(
            raw_root=raw_root,
            config=config,
            required_candidate_ids=required_candidates_by_dataset[dataset_name],
            image_paths=image_paths,
            batch_size=candidate_batch_size,
        )
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(CANDIDATE_SCHEMA, batches),
            str(output_root / "data" / "candidates" / f"{dataset_name}.lance"),
            schema=CANDIDATE_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
        candidate_counts[dataset_name] = lance.dataset(
            str(output_root / "data" / "candidates" / f"{dataset_name}.lance")
        ).count_rows()

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            IMAGE_SCHEMA,
            _image_batches(
                raw_root=raw_root,
                image_paths=image_paths,
                batch_size=image_batch_size,
                num_workers=num_workers,
            ),
        ),
        str(output_root / "data" / "images.lance"),
        schema=IMAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_mbeir_train_indices(output_root, selected_names)
    return {
        "output_root": str(output_root),
        "artifact_format": "mbeir_train_raw_lance_v1",
        "datasets": split_counts,
        "candidate_rows": candidate_counts,
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
    }


__all__ = [
    "CANDIDATE_SCHEMA",
    "IMAGE_SCHEMA",
    "MBEIR_TASK_CONFIGS",
    "MbeirTaskConfig",
    "QUERY_SCHEMA",
    "ensure_mbeir_train_indices",
    "write_mbeir_train_root",
]
