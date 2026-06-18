"""MSR-VTT training conversion utilities."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE
from vlm2emb.data.conversion.video_frame_units import (
    BAD_MEDIA_SCHEMA,
    FRAME_UNIT_SCHEMA,
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = (
    "train_9k",
    "train_7k",
    "test_1k",
    "train_9k_without_mmeb_v2_eval",
)

CAPTION_LIST_TYPE = pa.list_(pa.string())
VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("video_bytes", pa.binary()),
        pa.field("fps", pa.float32()),
        pa.field("total_num_frames", pa.int32()),
    ]
)
TRAIN_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("caption", CAPTION_LIST_TYPE),
        pa.field("source", pa.string()),
        pa.field("category", pa.int32()),
        pa.field("url", pa.string()),
        pa.field("start time", pa.float64()),
        pa.field("end time", pa.float64()),
        pa.field("id", pa.int64()),
    ]
)
TEST_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("source", pa.string()),
        pa.field("category", pa.int32()),
        pa.field("url", pa.string()),
        pa.field("start time", pa.float64()),
        pa.field("end time", pa.float64()),
        pa.field("id", pa.int64()),
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


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    """Load one MSR-VTT JSON split file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing MSR-VTT split file: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"MSR-VTT split file must contain a JSON list: {path}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"MSR-VTT row {index} in {path} must be an object.")
        normalized.append(dict(row))
    return normalized


def _normalize_train_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one train row while preserving the source field names."""

    captions = row.get("caption")
    if not isinstance(captions, list):
        raise TypeError(f"MSR-VTT train row caption must be a list: video_id={row.get('video_id')!r}")
    return {
        "video_id": str(row["video_id"]),
        "video": str(row["video"]),
        "caption": [str(caption) for caption in captions],
        "source": str(row.get("source", "")),
        "category": int(row["category"]),
        "url": str(row.get("url", "")),
        "start time": float(row["start time"]),
        "end time": float(row["end time"]),
        "id": int(row["id"]),
    }


def _normalize_test_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one test/eval row while preserving the source field names."""

    caption = row.get("caption")
    if isinstance(caption, list):
        caption = next((value for value in caption if isinstance(value, str) and value.strip()), "")
    return {
        "video_id": str(row["video_id"]),
        "video": str(row["video"]),
        "caption": str(caption or ""),
        "source": str(row.get("source", "")),
        "category": int(row["category"]),
        "url": str(row.get("url", "")),
        "start time": float(row["start time"]),
        "end time": float(row["end time"]),
        "id": int(row["id"]),
    }


def load_msrvtt_split_rows(source_root: Path) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return MSR-VTT train, eval, and eval-excluded train split rows."""

    train_9k = [_normalize_train_row(row) for row in _load_json_rows(source_root / "msrvtt_train_9k.json")]
    train_7k = [_normalize_train_row(row) for row in _load_json_rows(source_root / "msrvtt_train_7k.json")]
    test_1k = [_normalize_test_row(row) for row in _load_json_rows(source_root / "msrvtt_test_1k.json")]
    eval_video_ids = {row["video_id"] for row in test_1k}
    train_without_eval = [row for row in train_9k if row["video_id"] not in eval_video_ids]
    return OrderedDict(
        [
            ("train_9k", train_9k),
            ("train_7k", train_7k),
            ("test_1k", test_1k),
            ("train_9k_without_mmeb_v2_eval", train_without_eval),
        ]
    )


def collect_video_refs(source_root: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading video bytes."""

    video_id_by_path: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_msrvtt_split_rows(source_root).items():
        count = 0
        for row in rows:
            video_id_by_path.setdefault(str(row["video"]), str(row["video_id"]))
            count += 1
        counts[split_name] = count
    return video_id_by_path, counts


def _resolve_raw_video_path(source_root: Path, video: str) -> Path:
    """Resolve one MSR-VTT video path under the source root."""

    return source_root / "raw_videos" / video


def _read_raw_video_bytes(source_root: Path, video: str) -> bytes:
    """Read one MSR-VTT video from an extracted tree or the official zip."""

    raw_video_path = _resolve_raw_video_path(source_root, video)
    if raw_video_path.is_file():
        return raw_video_path.read_bytes()
    archive_path = source_root / "MSRVTT_Videos.zip"
    if archive_path.is_file():
        member = f"video/{video}"
        try:
            with zipfile.ZipFile(archive_path) as archive:
                return archive.read(member)
        except KeyError as error:
            raise FileNotFoundError(f"Missing raw MSR-VTT zip member for {video}: {archive_path}!{member}") from error
    raise FileNotFoundError(f"Missing raw MSR-VTT video for {video}: {raw_video_path}")


def _build_video_row(source_root: Path, item: tuple[str, str]) -> dict[str, Any]:
    """Read one raw video and return a Lance-ready row."""

    video, video_id = item
    video_bytes = _read_raw_video_bytes(source_root, video)
    metadata = probe_video_bytes(video_bytes, source_path=video)
    return {
        "video_id": video_id,
        "video": video,
        "video_bytes": video_bytes,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _video_batches(
    *,
    source_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build video table batches with bounded parallel file IO/probing."""

    workers = max(1, int(num_workers))
    items = list(video_refs.items())
    for chunk in _batched(items, batch_size):
        if workers == 1:
            rows = [_build_video_row(source_root, item) for item in chunk]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(lambda item: _build_video_row(source_root, item), chunk))
        yield pa.RecordBatch.from_pylist(rows, schema=VIDEOS_SCHEMA)


def _build_frame_unit_row(source_root: Path, item: tuple[str, str], *, max_frames: int) -> dict[str, Any]:
    """Read one raw clip and build a frame-unit row."""

    video, video_id = item
    return build_full_video_frame_unit_from_bytes(
        video_bytes=_read_raw_video_bytes(source_root, video),
        frame_unit_key=video,
        source_key=video_id,
        source_path=video,
        max_frames=max_frames,
    )


def _frame_unit_batches(
    *,
    source_root: Path,
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
                    results.append((item, _build_frame_unit_row(source_root, item, max_frames=full_video_max_frames)))
                except Exception as error:
                    results.append((item, error))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    (item, executor.submit(_build_frame_unit_row, source_root, item, max_frames=full_video_max_frames))
                    for item in chunk
                ]
                for item, future in futures:
                    try:
                        results.append((item, future.result()))
                    except Exception as error:
                        results.append((item, error))

        rows: list[dict[str, Any]] = []
        for item, result in results:
            video, video_id = item
            if isinstance(result, Exception):
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="MSR-VTT",
                        split="all",
                        source_key=video_id,
                        frame_unit_key=video,
                        reason="invalid_video",
                        error=str(result),
                        source_path=video,
                    )
                )
                continue
            rows.append(result)
            valid_keys.add(video)
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


def _schema_for_split(split_name: str) -> pa.Schema:
    """Return the source-preserving Arrow schema for one split."""

    return TEST_SPLIT_SCHEMA if split_name == "test_1k" else TRAIN_SPLIT_SCHEMA


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_msrvtt_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the MSR-VTT runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing MSR-VTT frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video")


def write_msrvtt_root(
    *,
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    full_video_max_frames: int = 64,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert MSR-VTT into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    video_refs, split_counts = collect_video_refs(source_root)
    valid_video_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                source_root=source_root,
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
        (video, video_id)
        for video, video_id in video_refs.items()
        if video in valid_video_keys
    )
    video_reader = pa.RecordBatchReader.from_batches(
        VIDEOS_SCHEMA,
        _video_batches(
            source_root=source_root,
            video_refs=valid_video_refs,
            batch_size=video_batch_size,
            num_workers=num_workers,
        ),
    )
    lance.write_dataset(
        video_reader,
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_msrvtt_split_rows(source_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video"] in valid_video_keys]
        split_counts[split_name] = len(filtered_rows)
        schema = _schema_for_split(split_name)
        reader = pa.RecordBatchReader.from_batches(
            schema,
            _split_batches(filtered_rows, schema=schema, batch_size=split_batch_size),
        )
        lance.write_dataset(
            reader,
            str(output_root / "data" / f"{split_name}.lance"),
            schema=schema,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    exclusions = split_rows["test_1k"]
    exclusion_reader = pa.RecordBatchReader.from_batches(
        TEST_SPLIT_SCHEMA,
        _split_batches(exclusions, schema=TEST_SPLIT_SCHEMA, batch_size=split_batch_size),
    )
    lance.write_dataset(
        exclusion_reader,
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=TEST_SPLIT_SCHEMA,
        mode="create",
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

    ensure_msrvtt_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "splits": split_counts,
        "exclusion_rows": split_counts["test_1k"],
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "msrvtt_train_frame_unit_lance_v1",
    }


__all__ = [
    "SUPPORTED_SPLITS",
    "TEST_SPLIT_SCHEMA",
    "TRAIN_SPLIT_SCHEMA",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_msrvtt_indices",
    "load_msrvtt_split_rows",
    "write_msrvtt_root",
]
