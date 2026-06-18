"""Charades-STA training conversion utilities."""

from __future__ import annotations

import shutil
import threading
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
    FULL_VIDEO_MAX_FRAMES,
    SEGMENT_MAX_FRAMES,
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
    build_segment_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = ("official_train", "official_test", "official_train_without_mmeb_v2_eval")

VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("video_bytes", pa.binary()),
        pa.field("fps", pa.float32()),
        pa.field("total_num_frames", pa.int32()),
    ]
)
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("start_time", pa.float32()),
        pa.field("end_time", pa.float32()),
        pa.field("duration", pa.float32()),
        pa.field("description", pa.string()),
        pa.field("query_frame_unit_key", pa.string()),
        pa.field("positive_frame_unit_key", pa.string()),
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


def _parse_annotation_line(line: str, *, path: Path, line_number: int) -> dict[str, Any]:
    """Parse one official Charades-STA annotation line."""

    try:
        prefix, description = line.split("##", maxsplit=1)
        video_id, start_time, end_time = prefix.split()
    except ValueError as exc:
        raise ValueError(f"Invalid Charades-STA annotation line {line_number} in {path}: {line!r}") from exc
    start = float(start_time)
    end = float(end_time)
    return {
        "id": video_id.strip(),
        "start_time": start,
        "end_time": end,
        "duration": round(end - start, 1),
        "description": description.strip(),
        "query_frame_unit_key": _query_frame_unit_key(video_id.strip()),
        "positive_frame_unit_key": _segment_frame_unit_key(video_id.strip(), start, end),
    }


def _load_annotation_rows(path: Path) -> list[dict[str, Any]]:
    """Load one Charades-STA annotation split."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing Charades-STA annotation file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        rows.append(_parse_annotation_line(line, path=path, line_number=line_number))
    return rows


def load_charades_sta_split_rows(raw_root: Path) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return official train/test and eval-excluded train split rows."""

    annotation_root = raw_root / "annotations" / "charades_sta"
    train = _load_annotation_rows(annotation_root / "Charades_sta_train.txt")
    test = _load_annotation_rows(annotation_root / "Charades_sta_test.txt")
    eval_keys = {(row["id"], row["start_time"], row["end_time"], row["description"]) for row in test}
    train_without_eval = [
        row
        for row in train
        if (row["id"], row["start_time"], row["end_time"], row["description"]) not in eval_keys
    ]
    return OrderedDict(
        [
            ("official_train", train),
            ("official_test", test),
            ("official_train_without_mmeb_v2_eval", train_without_eval),
        ]
    )


def collect_video_refs(raw_root: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique Charades video refs and split row counts."""

    video_by_id: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_charades_sta_split_rows(raw_root).items():
        count = 0
        for row in rows:
            video_id = str(row["id"])
            video_by_id.setdefault(video_id, f"Charades_v1/{video_id}.mp4")
            count += 1
        counts[split_name] = count
    return video_by_id, counts


def _query_frame_unit_key(video_id: str) -> str:
    """Build the query-side full-video frame-unit key."""

    return f"{video_id}#full"


def _segment_frame_unit_key(video_id: str, start_time: float, end_time: float) -> str:
    """Build the positive-side segment frame-unit key."""

    return f"{video_id}#segment={float(start_time):.3f}-{float(end_time):.3f}"


def _segments_by_video(raw_root: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Collect unique Charades-STA segment rows by source video id."""

    grouped: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
    for rows in load_charades_sta_split_rows(raw_root).values():
        for row in rows:
            video_id = str(row["id"])
            grouped.setdefault(video_id, OrderedDict())
            grouped[video_id].setdefault(str(row["positive_frame_unit_key"]), row)
    return OrderedDict((video_id, list(rows.values())) for video_id, rows in grouped.items())


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_charades_sta_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Charades-STA runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing Charades-STA frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_id")


def _zip_reader(zip_path: Path, video: str) -> bytes:
    """Read one video member using a thread-local ZipFile handle."""

    local = _zip_reader_state
    handle = getattr(local, "handle", None)
    handle_path = getattr(local, "path", None)
    if handle is None or handle_path != zip_path:
        if handle is not None:
            handle.close()
        handle = zipfile.ZipFile(zip_path)
        local.handle = handle
        local.path = zip_path
    try:
        return handle.read(video)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing Charades video member '{video}' in archive {zip_path}") from exc


_zip_reader_state = threading.local()


def _build_video_row(zip_path: Path, video_id: str, video: str) -> dict[str, Any]:
    """Build one Lance-ready video row from a zip member."""

    video_bytes = _zip_reader(zip_path, video)
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
    zip_path: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build ordered video batches from the Charades zip archive."""

    items = list(video_refs.items())
    workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = executor.map(lambda item: _build_video_row(zip_path, item[0], item[1]), items)
        for batch in _batched(rows, batch_size):
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)


def _build_frame_unit_rows(
    zip_path: Path,
    video_id: str,
    video: str,
    segments: list[dict[str, Any]],
    *,
    full_video_max_frames: int,
    segment_max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build mixed-view frame-units for one Charades-STA source video."""

    rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    try:
        video_bytes = _zip_reader(zip_path, video)
    except Exception as error:
        bad_rows.append(
            build_bad_media_row(
                dataset="Charades-STA",
                split="all",
                source_key=video_id,
                frame_unit_key=_query_frame_unit_key(video_id),
                reason="missing_video",
                error=str(error),
                source_path=video,
            )
        )
        return rows, bad_rows

    try:
        rows.append(
            build_full_video_frame_unit_from_bytes(
                video_bytes=video_bytes,
                frame_unit_key=_query_frame_unit_key(video_id),
                source_key=video_id,
                source_path=video,
                max_frames=full_video_max_frames,
            )
        )
    except Exception as error:
        bad_rows.append(
            build_bad_media_row(
                dataset="Charades-STA",
                split="all",
                source_key=video_id,
                frame_unit_key=_query_frame_unit_key(video_id),
                reason="invalid_video",
                error=str(error),
                source_path=video,
            )
        )

    for segment in segments:
        frame_unit_key = str(segment["positive_frame_unit_key"])
        try:
            rows.append(
                build_segment_frame_unit_from_bytes(
                    video_bytes=video_bytes,
                    frame_unit_key=frame_unit_key,
                    source_key=video_id,
                    source_path=video,
                    start_time=float(segment["start_time"]),
                    end_time=float(segment["end_time"]),
                    max_frames=segment_max_frames,
                )
            )
        except Exception as error:
            bad_rows.append(
                build_bad_media_row(
                    dataset="Charades-STA",
                    split="all",
                    source_key=video_id,
                    frame_unit_key=frame_unit_key,
                    reason="invalid_segment",
                    error=str(error),
                    source_path=video,
                )
            )
    return rows, bad_rows


def _frame_unit_batches(
    *,
    zip_path: Path,
    video_refs: OrderedDict[str, str],
    segments_by_video: OrderedDict[str, list[dict[str, Any]]],
    batch_size: int,
    num_workers: int,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
    segment_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build mixed-view frame-unit batches from the Charades zip archive."""

    items = list(video_refs.items())
    workers = max(1, int(num_workers))
    pending_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda item: _build_frame_unit_rows(
                zip_path,
                item[0],
                item[1],
                segments_by_video.get(item[0], []),
                full_video_max_frames=full_video_max_frames,
                segment_max_frames=segment_max_frames,
            ),
            items,
        )
        for rows, bad_rows in results:
            bad_media_rows.extend(bad_rows)
            for row in rows:
                valid_keys.add(str(row["frame_unit_key"]))
                pending_rows.append(row)
                if len(pending_rows) >= batch_size:
                    yield pa.RecordBatch.from_pylist(pending_rows, schema=FRAME_UNIT_SCHEMA)
                    pending_rows = []
    if pending_rows:
        yield pa.RecordBatch.from_pylist(pending_rows, schema=FRAME_UNIT_SCHEMA)


def _bad_media_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield bad-media rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=BAD_MEDIA_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=BAD_MEDIA_SCHEMA)


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)


def write_charades_sta_root(
    *,
    raw_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    segment_max_frames: int = SEGMENT_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert Charades-STA into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    zip_path = raw_root / "videos" / "Charades_v1.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing Charades video archive: {zip_path}")

    video_refs, split_counts = collect_video_refs(raw_root)
    segments_by_video = _segments_by_video(raw_root)
    valid_frame_unit_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                zip_path=zip_path,
                video_refs=video_refs,
                segments_by_video=segments_by_video,
                batch_size=video_batch_size,
                num_workers=num_workers,
                valid_keys=valid_frame_unit_keys,
                bad_media_rows=bad_media_rows,
                full_video_max_frames=full_video_max_frames,
                segment_max_frames=segment_max_frames,
            ),
        ),
        str(output_root / "data" / "frames.lance"),
        schema=FRAME_UNIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    valid_video_refs = OrderedDict(
        (video_id, video)
        for video_id, video in video_refs.items()
        if _query_frame_unit_key(video_id) in valid_frame_unit_keys
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches(
                zip_path=zip_path,
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

    split_rows = load_charades_sta_split_rows(raw_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [
            row
            for row in rows
            if row["query_frame_unit_key"] in valid_frame_unit_keys
            and row["positive_frame_unit_key"] in valid_frame_unit_keys
        ]
        split_counts[split_name] = len(filtered_rows)
        reader = pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches(filtered_rows, batch_size=split_batch_size),
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
        _split_batches(split_rows["official_test"], batch_size=split_batch_size),
    )
    lance.write_dataset(
        exclusion_reader,
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
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

    ensure_charades_sta_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_frame_unit_keys),
        "splits": split_counts,
        "exclusion_rows": split_counts["official_test"],
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "charades_sta_train_frame_unit_lance_v1",
    }


__all__ = [
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_charades_sta_indices",
    "load_charades_sta_split_rows",
    "write_charades_sta_root",
]
