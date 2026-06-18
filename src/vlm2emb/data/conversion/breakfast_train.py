"""Breakfast training conversion utilities."""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE
from vlm2emb.data.conversion.video_frame_units import (
    BAD_MEDIA_SCHEMA,
    FRAME_UNIT_SCHEMA,
    FULL_VIDEO_MAX_FRAMES,
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = (
    "official_s1",
    "official_s2",
    "official_s3",
    "official_s4",
    "archive_train_non_cam01",
    "vlm2vec_test_cam01_433",
)
LABEL_ALIASES = {
    "cereals": "cereal",
    "salat": "salad",
}

LABELS_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("start", pa.int64()),
            pa.field("end", pa.int64()),
            pa.field("label", pa.string()),
        ]
    )
)
VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video_path", pa.string()),
        pa.field("video", pa.binary()),
        pa.field("fps", pa.float32()),
        pa.field("total_num_frames", pa.int32()),
    ]
)
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video_path", pa.string()),
        pa.field("label", pa.string()),
        pa.field("raw_label", pa.string()),
        pa.field("participant", pa.string()),
        pa.field("camera", pa.string()),
        pa.field("source_split", pa.string()),
        pa.field("labels", LABELS_TYPE),
    ]
)
EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video_path", pa.string()),
        pa.field("pos_text", pa.string()),
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


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL objects from one local file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL row in {path} at line {line_number}.") from error
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row in {path} at line {line_number} must be an object.")
            yield row


def _raw_label_from_video_name(video_name: str) -> str:
    """Infer the raw Breakfast class label from the upstream video name."""

    parts = video_name.split("_")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Cannot infer Breakfast label from video name: {video_name}")
    return parts[1]


def _canonical_label(raw_label: str) -> str:
    """Map raw Breakfast filename labels to eval-candidate labels."""

    return LABEL_ALIASES.get(raw_label, raw_label)


def _normalize_video_path(video_path: str) -> str:
    """Normalize upstream/MMEB paths for overlap comparisons."""

    normalized = video_path.removeprefix("videos/")
    for source_label, target_label in (("cereal", "cereals"), ("salad", "salat")):
        normalized = normalized.replace(f"_{source_label}.avi", f"_{target_label}.avi")
    return normalized


def _normalize_video_id(video_id: str) -> str:
    """Normalize upstream/MMEB video ids for overlap comparisons."""

    for source_label, target_label in (("cereal", "cereals"), ("salad", "salat")):
        if video_id.endswith(f"_{source_label}"):
            return video_id[: -len(source_label)] + target_label
    return video_id


def _load_official_split_rows(raw_root: Path, split_name: str) -> list[dict[str, Any]]:
    """Read one official Breakfast participant split."""

    parquet_path = raw_root / "data" / f"{split_name}-00000-of-00001.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing Breakfast split parquet: {parquet_path}")
    table = pq.read_table(parquet_path, columns=["video_path", "participant", "camera", "video", "labels"])
    rows: list[dict[str, Any]] = []
    for row in table.to_pylist():
        video_path = str(row["video_path"])
        video_id = str(row["video"])
        raw_label = _raw_label_from_video_name(video_id)
        rows.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "label": _canonical_label(raw_label),
                "raw_label": raw_label,
                "participant": str(row["participant"]),
                "camera": str(row["camera"]),
                "source_split": split_name,
                "labels": row.get("labels") or [],
            }
        )
    return rows


def load_mmeb_v2_eval_exclusion_rows(vlm2vec_root: Path) -> list[dict[str, str]]:
    """Load the VLM2Vec MMEB-V2 eval view used for Breakfast exclusion."""

    test_jsonl = vlm2vec_root / "data" / "test.jsonl"
    if not test_jsonl.is_file():
        raise FileNotFoundError(f"Missing Breakfast VLM2Vec eval metadata: {test_jsonl}")
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for row in _read_jsonl(test_jsonl):
        video_path = _normalize_video_path(str(row["video_path"]))
        if video_path in seen_paths:
            continue
        seen_paths.add(video_path)
        rows.append(
            {
                "video_id": _normalize_video_id(str(row["video_id"])),
                "video_path": video_path,
                "pos_text": str(row["pos_text"]),
            }
        )
    return rows


def load_breakfast_split_rows(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return split row iterables for official, archive train, and eval views."""

    split_rows: OrderedDict[str, Iterable[dict[str, Any]]] = OrderedDict()
    archive_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for split_name in ("s1", "s2", "s3", "s4"):
        rows = _load_official_split_rows(raw_root, split_name)
        split_rows[f"official_{split_name}"] = rows
        archive_rows.extend(row for row in rows if row["camera"] != "cam01")
        eval_rows.extend(row for row in rows if row["camera"] == "cam01")

    exclusions = load_mmeb_v2_eval_exclusion_rows(vlm2vec_root)
    eval_paths = {_normalize_video_path(row["video_path"]) for row in eval_rows}
    missing_eval_paths = [row["video_path"] for row in exclusions if row["video_path"] not in eval_paths]
    if missing_eval_paths:
        preview = ", ".join(missing_eval_paths[:5])
        raise ValueError(f"Breakfast MMEB-V2 eval rows are missing from official cam01 split: {preview}")

    split_rows["archive_train_non_cam01"] = archive_rows
    split_rows["vlm2vec_test_cam01_433"] = eval_rows
    return split_rows


def collect_video_refs(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading video bytes."""

    video_id_by_path: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_breakfast_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root).items():
        count = 0
        for row in rows:
            video_path = str(row["video_path"])
            video_id_by_path.setdefault(video_path, str(row["video_id"]))
            count += 1
        counts[split_name] = count
    return video_id_by_path, counts


def _resolve_raw_video_path(raw_root: Path, relative_video_path: str) -> Path:
    """Resolve one Breakfast video path under the raw root."""

    return raw_root / relative_video_path


def _build_video_row(raw_root: Path, item: tuple[str, str]) -> dict[str, Any]:
    """Read one raw video and return a Lance-ready row."""

    video_path, video_id = item
    raw_video_path = _resolve_raw_video_path(raw_root, video_path)
    if not raw_video_path.is_file():
        raise FileNotFoundError(f"Missing raw Breakfast video for {video_path}: {raw_video_path}")
    video = raw_video_path.read_bytes()
    metadata = probe_video_bytes(video, source_path=video_path)
    return {
        "video_id": video_id,
        "video_path": video_path,
        "video": video,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _video_batches(
    *,
    raw_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build video table batches with bounded parallel file IO/probing."""

    workers = max(1, int(num_workers))
    items = list(video_refs.items())
    for chunk in _batched(items, batch_size):
        if workers == 1:
            rows = [_build_video_row(raw_root, item) for item in chunk]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(lambda item: _build_video_row(raw_root, item), chunk))
        yield pa.RecordBatch.from_pylist(rows, schema=VIDEOS_SCHEMA)


def _build_frame_unit_row(raw_root: Path, item: tuple[str, str], *, max_frames: int) -> dict[str, Any]:
    """Read one raw video and build a frame-unit row."""

    video_path, video_id = item
    raw_video_path = _resolve_raw_video_path(raw_root, video_path)
    if not raw_video_path.is_file():
        raise FileNotFoundError(f"Missing raw Breakfast video for {video_path}: {raw_video_path}")
    return build_full_video_frame_unit_from_bytes(
        video_bytes=raw_video_path.read_bytes(),
        frame_unit_key=video_path,
        source_key=video_id,
        source_path=video_path,
        max_frames=max_frames,
    )


def _frame_unit_batches(
    *,
    raw_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build frame-unit batches while collecting bad media."""

    workers = max(1, int(num_workers))
    items = list(video_refs.items())
    for chunk in _batched(items, batch_size):
        results: list[tuple[tuple[str, str], dict[str, Any] | Exception]] = []
        if workers == 1:
            for item in chunk:
                try:
                    results.append((item, _build_frame_unit_row(raw_root, item, max_frames=full_video_max_frames)))
                except Exception as error:
                    results.append((item, error))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    (item, executor.submit(_build_frame_unit_row, raw_root, item, max_frames=full_video_max_frames))
                    for item in chunk
                ]
                for item, future in futures:
                    try:
                        results.append((item, future.result()))
                    except Exception as error:
                        results.append((item, error))
        rows: list[dict[str, Any]] = []
        for item, result in results:
            video_path, video_id = item
            if isinstance(result, Exception):
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="Breakfast",
                        split="all",
                        source_key=video_id,
                        frame_unit_key=video_path,
                        reason="invalid_video",
                        error=str(result),
                        source_path=video_path,
                    )
                )
                continue
            rows.append(result)
            valid_keys.add(video_path)
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=FRAME_UNIT_SCHEMA)


def _bad_media_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield bad-media rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=BAD_MEDIA_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=BAD_MEDIA_SCHEMA)


def _split_batches(rows: Iterable[dict[str, Any]], *, schema: pa.Schema, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=schema)


def _write_rows(path: Path, rows: list[dict[str, Any]], *, schema: pa.Schema, max_bytes_per_file: int) -> None:
    """Write a small in-memory row set while preserving empty-table schemas."""

    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=schema),
        str(path),
        schema=schema,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_breakfast_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Breakfast runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing Breakfast frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_path")


def write_breakfast_root(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 32,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert Breakfast into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    exclusions = load_mmeb_v2_eval_exclusion_rows(vlm2vec_root)
    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    valid_video_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                raw_root=raw_root,
                video_refs=video_refs,
                batch_size=video_batch_size,
                num_workers=num_workers,
                valid_keys=valid_video_keys,
                bad_media_rows=bad_media_rows,
                full_video_max_frames=full_video_max_frames,
            ),
        ),
        str(output_root / "data" / "frames.lance"),
        schema=FRAME_UNIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    valid_video_refs = OrderedDict(
        (video_path, video_id)
        for video_path, video_id in video_refs.items()
        if video_path in valid_video_keys
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches(
                raw_root=raw_root,
                video_refs=valid_video_refs,
                batch_size=video_batch_size,
                num_workers=num_workers,
            ),
        ),
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_breakfast_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video_path"] in valid_video_keys]
        split_counts[split_name] = len(filtered_rows)
        reader = pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches(filtered_rows, schema=SPLIT_SCHEMA, batch_size=split_batch_size),
        )
        lance.write_dataset(
            reader,
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    _write_rows(
        output_root / "data" / "exclusions" / "mmeb_v2_eval.lance",
        exclusions,
        schema=EXCLUSION_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            BAD_MEDIA_SCHEMA,
            _bad_media_batches(bad_media_rows, batch_size=split_batch_size),
        ),
        str(output_root / "data" / "exclusions" / "bad_media.lance"),
        schema=BAD_MEDIA_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_breakfast_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "exclusion_rows": len(exclusions),
        "splits": split_counts,
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "breakfast_train_frame_unit_lance_v1",
    }


__all__ = [
    "EXCLUSION_SCHEMA",
    "LABELS_TYPE",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_breakfast_indices",
    "load_breakfast_split_rows",
    "load_mmeb_v2_eval_exclusion_rows",
    "write_breakfast_root",
]
