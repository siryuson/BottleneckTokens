"""Kinetics-700 training conversion utilities."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import queue
import shutil
import tarfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

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
    "official_train",
    "train_without_mmeb_v2_eval",
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
TRAIN_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video_path", pa.string()),
        pa.field("label", pa.string()),
        pa.field("youtube_id", pa.string()),
        pa.field("time_start", pa.int32()),
        pa.field("time_end", pa.int32()),
        pa.field("split", pa.string()),
    ]
)
EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video_path", pa.string()),
        pa.field("pos_text", pa.string()),
    ]
)
INVALID_VIDEO_SCHEMA = pa.schema(
    [
        pa.field("video_path", pa.string()),
        pa.field("error", pa.string()),
    ]
)


@dataclass(frozen=True)
class _WorkerError:
    """Exception raised inside a tar-streaming worker."""

    error: BaseException


@dataclass(frozen=True)
class _InvalidVideo:
    """Invalid video reported by a tar-streaming worker."""

    video_path: str
    error: str


_DONE = object()


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


def _format_video_id(youtube_id: str, time_start: int, time_end: int) -> str:
    """Build the official Kinetics video id used by tar members."""

    return f"{youtube_id}_{time_start:06d}_{time_end:06d}"


def _normalize_official_train_row(row: dict[str, str], *, source_path: Path, line_number: int) -> dict[str, Any]:
    """Normalize one official Kinetics-700 train CSV row."""

    try:
        label = row["label"].strip()
        youtube_id = row["youtube_id"].strip()
        time_start = int(row["time_start"])
        time_end = int(row["time_end"])
        split = row["split"].strip()
    except KeyError as error:
        raise ValueError(f"Missing Kinetics-700 CSV column {error.args[0]!r} in {source_path}.") from error
    except ValueError as error:
        raise ValueError(f"Invalid Kinetics-700 timestamp in {source_path} at row {line_number}.") from error
    if not label or not youtube_id:
        raise ValueError(f"Empty Kinetics-700 label/youtube_id in {source_path} at row {line_number}.")
    video_id = _format_video_id(youtube_id, time_start, time_end)
    return {
        "video_id": video_id,
        "video_path": f"{label}/{video_id}.mp4",
        "label": label,
        "youtube_id": youtube_id,
        "time_start": time_start,
        "time_end": time_end,
        "split": split,
    }


def _load_official_train_rows(raw_root: Path) -> list[dict[str, Any]]:
    """Read official train CSV rows and deduplicate repeated video keys."""

    train_csv = raw_root / "annotations" / "train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Missing Kinetics-700 train annotations: {train_csv}")

    rows_by_path: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            normalized = _normalize_official_train_row(row, source_path=train_csv, line_number=line_number)
            rows_by_path.setdefault(normalized["video_path"], normalized)
    return list(rows_by_path.values())


def load_mmeb_v2_eval_exclusion_rows(vlm2vec_root: Path) -> list[dict[str, str]]:
    """Load the VLM2Vec MMEB-V2 eval view used for Kinetics-700 exclusion."""

    test_jsonl = vlm2vec_root / "data" / "test.jsonl"
    if not test_jsonl.is_file():
        raise FileNotFoundError(f"Missing Kinetics-700 VLM2Vec eval metadata: {test_jsonl}")
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for row in _read_jsonl(test_jsonl):
        video_path = str(row["video_path"])
        if video_path in seen_paths:
            continue
        seen_paths.add(video_path)
        rows.append(
            {
                "video_id": str(row["video_id"]),
                "video_path": video_path,
                "pos_text": str(row["pos_text"]),
            }
        )
    return rows


def _exclude_eval_rows(
    train_rows: list[dict[str, Any]],
    exclusion_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Remove rows that match MMEB-V2 eval by video_path or video_id."""

    excluded_paths = {row["video_path"] for row in exclusion_rows}
    excluded_ids = {row["video_id"] for row in exclusion_rows}
    return [
        row
        for row in train_rows
        if row["video_path"] not in excluded_paths and row["video_id"] not in excluded_ids
    ]


def load_kinetics700_split_rows(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return split row iterables for official and MMEB-V2-excluded train views."""

    official_train = _load_official_train_rows(raw_root)
    exclusions = load_mmeb_v2_eval_exclusion_rows(vlm2vec_root)
    return OrderedDict(
        [
            ("official_train", official_train),
            ("train_without_mmeb_v2_eval", _exclude_eval_rows(official_train, exclusions)),
        ]
    )


def collect_video_refs(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading video bytes."""

    video_id_by_path: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_kinetics700_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root).items():
        count = 0
        for row in rows:
            video_path = str(row["video_path"])
            video_id_by_path.setdefault(video_path, str(row["video_id"]))
            count += 1
        counts[split_name] = count
    return video_id_by_path, counts


def _tar_paths(raw_root: Path) -> list[Path]:
    """Return official Kinetics-700 train tar files in stable order."""

    train_tars = raw_root / "train_tars"
    if not train_tars.is_dir():
        raise FileNotFoundError(f"Missing Kinetics-700 train tar directory: {train_tars}")
    paths = sorted(train_tars.glob("*.tar.gz"))
    if not paths:
        raise FileNotFoundError(f"No Kinetics-700 train tar files found under {train_tars}")
    return paths


def _build_video_row(*, video_id: str, video_path: str, video: bytes) -> dict[str, Any]:
    """Build one Lance-ready video row from raw video bytes."""

    try:
        metadata = probe_video_bytes(video, source_path=video_path)
    except Exception as error:
        raise ValueError(f"Failed to probe Kinetics-700 video bytes for {video_path}.") from error
    return {
        "video_id": video_id,
        "video_path": video_path,
        "video": video,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _build_frame_unit_row(
    *,
    video_id: str,
    video_path: str,
    video: bytes,
    max_frames: int,
) -> dict[str, Any]:
    """Build one full-video frame-unit row from raw video bytes."""

    try:
        return build_full_video_frame_unit_from_bytes(
            video_bytes=video,
            frame_unit_key=video_path,
            source_key=video_id,
            source_path=video_path,
            max_frames=max_frames,
        )
    except Exception as error:
        raise ValueError(f"Failed to sample Kinetics-700 video frames for {video_path}.") from error


def _emit_progress(event: str, **payload: Any) -> None:
    """Print one machine-readable progress event."""

    print(
        json.dumps(
            {
                "event": event,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "pid": os.getpid(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _scan_tar_worker(
    *,
    paths: list[Path],
    video_id_by_path: dict[str, str],
    output_queue: queue.Queue[object],
    batch_size: int,
    stop_event: threading.Event,
    invalid_video_policy: str,
    progress_label: str | None = None,
    shard_index: int | None = None,
) -> None:
    """Stream selected Kinetics-700 videos from tar files into row batches."""

    batch: list[dict[str, Any]] = []
    try:
        for tar_path in paths:
            if stop_event.is_set():
                break
            tar_started_at = time.monotonic()
            tar_rows = 0
            tar_invalid = 0
            if progress_label is not None:
                _emit_progress(
                    "kinetics700_tar_start",
                    phase=progress_label,
                    shard_index=shard_index,
                    tar=str(tar_path),
                )
            with tarfile.open(tar_path, mode="r:*") as tar:
                for member in tar:
                    if stop_event.is_set():
                        break
                    if not member.isfile():
                        continue
                    video_path = member.name.lstrip("./")
                    video_id = video_id_by_path.get(video_path)
                    if video_id is None:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise FileNotFoundError(f"Cannot extract {video_path} from {tar_path}")
                    video = extracted.read()
                    try:
                        batch.append(_build_video_row(video_id=video_id, video_path=video_path, video=video))
                    except Exception as error:
                        if invalid_video_policy != "skip":
                            raise
                        output_queue.put(_InvalidVideo(video_path=video_path, error=str(error)))
                        tar_invalid += 1
                        continue
                    tar_rows += 1
                    if len(batch) >= batch_size:
                        output_queue.put(batch)
                        batch = []
            if progress_label is not None:
                _emit_progress(
                    "kinetics700_tar_done",
                    phase=progress_label,
                    shard_index=shard_index,
                    tar=str(tar_path),
                    rows=tar_rows,
                    invalid_rows=tar_invalid,
                    elapsed_sec=round(time.monotonic() - tar_started_at, 3),
                )
        if batch:
            output_queue.put(batch)
    except BaseException as error:
        stop_event.set()
        output_queue.put(_WorkerError(error))
    finally:
        output_queue.put(_DONE)


def _scan_tar_frame_unit_worker(
    *,
    paths: list[Path],
    video_id_by_path: dict[str, str],
    output_queue: queue.Queue[object],
    batch_size: int,
    stop_event: threading.Event,
    invalid_video_policy: str,
    full_video_max_frames: int,
    progress_label: str | None = None,
    shard_index: int | None = None,
) -> None:
    """Stream selected Kinetics-700 videos from tar files into frame-unit batches."""

    batch: list[dict[str, Any]] = []
    try:
        for tar_path in paths:
            if stop_event.is_set():
                break
            tar_started_at = time.monotonic()
            tar_rows = 0
            tar_invalid = 0
            if progress_label is not None:
                _emit_progress(
                    "kinetics700_tar_start",
                    phase=progress_label,
                    shard_index=shard_index,
                    tar=str(tar_path),
                )
            with tarfile.open(tar_path, mode="r:*") as tar:
                for member in tar:
                    if stop_event.is_set():
                        break
                    if not member.isfile():
                        continue
                    video_path = member.name.lstrip("./")
                    video_id = video_id_by_path.get(video_path)
                    if video_id is None:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise FileNotFoundError(f"Cannot extract {video_path} from {tar_path}")
                    video = extracted.read()
                    try:
                        batch.append(
                            _build_frame_unit_row(
                                video_id=video_id,
                                video_path=video_path,
                                video=video,
                                max_frames=full_video_max_frames,
                            )
                        )
                    except Exception as error:
                        if invalid_video_policy != "skip":
                            raise
                        output_queue.put(_InvalidVideo(video_path=video_path, error=str(error)))
                        tar_invalid += 1
                        continue
                    tar_rows += 1
                    if len(batch) >= batch_size:
                        output_queue.put(batch)
                        batch = []
            if progress_label is not None:
                _emit_progress(
                    "kinetics700_tar_done",
                    phase=progress_label,
                    shard_index=shard_index,
                    tar=str(tar_path),
                    rows=tar_rows,
                    invalid_rows=tar_invalid,
                    elapsed_sec=round(time.monotonic() - tar_started_at, 3),
                )
        if batch:
            output_queue.put(batch)
    except BaseException as error:
        stop_event.set()
        output_queue.put(_WorkerError(error))
    finally:
        output_queue.put(_DONE)


def _video_batches_from_tars(
    *,
    raw_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
    invalid_video_policy: str,
    invalid_video_rows: list[dict[str, str]],
) -> Iterator[pa.RecordBatch]:
    """Build video table batches by streaming the official compressed train tars."""

    if invalid_video_policy not in {"error", "skip"}:
        raise ValueError("invalid_video_policy must be 'error' or 'skip'.")
    if not video_refs:
        return

    tar_paths = _tar_paths(raw_root)
    workers = max(1, min(int(num_workers), len(tar_paths)))
    path_groups = [tar_paths[index::workers] for index in range(workers)]
    video_id_by_path = dict(video_refs)
    output_queue: queue.Queue[object] = queue.Queue(maxsize=max(2, workers * 2))
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_scan_tar_worker,
            kwargs={
                "paths": group,
                "video_id_by_path": video_id_by_path,
                "output_queue": output_queue,
                "batch_size": batch_size,
                "stop_event": stop_event,
                "invalid_video_policy": invalid_video_policy,
            },
            daemon=True,
        )
        for group in path_groups
        if group
    ]
    for thread in threads:
        thread.start()

    seen_paths: set[str] = set()
    invalid_by_path: OrderedDict[str, str] = OrderedDict()
    done_workers = 0
    pending_error: BaseException | None = None
    try:
        while done_workers < len(threads):
            item = output_queue.get()
            if item is _DONE:
                done_workers += 1
                continue
            if isinstance(item, _WorkerError):
                stop_event.set()
                pending_error = item.error
                continue
            if isinstance(item, _InvalidVideo):
                if item.video_path not in seen_paths:
                    invalid_by_path.setdefault(item.video_path, item.error)
                continue
            rows = item
            if not isinstance(rows, list):
                raise TypeError(f"Unexpected Kinetics-700 worker output: {type(rows).__name__}")
            if pending_error is not None:
                continue
            deduplicated_rows: list[dict[str, Any]] = []
            for row in rows:
                video_path = row["video_path"]
                if video_path in seen_paths:
                    continue
                seen_paths.add(video_path)
                invalid_by_path.pop(video_path, None)
                deduplicated_rows.append(row)
            if deduplicated_rows:
                yield pa.RecordBatch.from_pylist(deduplicated_rows, schema=VIDEOS_SCHEMA)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    invalid_video_rows.extend(
        {"video_path": video_path, "error": error}
        for video_path, error in invalid_by_path.items()
    )
    missing_paths = [path for path in video_refs if path not in seen_paths and path not in invalid_by_path]
    if pending_error is not None:
        raise pending_error
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        raise FileNotFoundError(
            f"Kinetics-700 train tars are missing {len(missing_paths)} referenced videos. "
            f"First missing paths: {preview}"
        )


def _frame_unit_batches_from_tars(
    *,
    raw_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
    invalid_video_policy: str,
    invalid_video_rows: list[dict[str, str]],
    valid_keys: set[str],
    full_video_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build frame-unit batches by streaming the official compressed train tars."""

    if invalid_video_policy not in {"error", "skip"}:
        raise ValueError("invalid_video_policy must be 'error' or 'skip'.")
    if not video_refs:
        return

    tar_paths = _tar_paths(raw_root)
    workers = max(1, min(int(num_workers), len(tar_paths)))
    path_groups = [tar_paths[index::workers] for index in range(workers)]
    video_id_by_path = dict(video_refs)
    output_queue: queue.Queue[object] = queue.Queue(maxsize=max(2, workers * 2))
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_scan_tar_frame_unit_worker,
            kwargs={
                "paths": group,
                "video_id_by_path": video_id_by_path,
                "output_queue": output_queue,
                "batch_size": batch_size,
                "stop_event": stop_event,
                "invalid_video_policy": invalid_video_policy,
                "full_video_max_frames": full_video_max_frames,
            },
            daemon=True,
        )
        for group in path_groups
        if group
    ]
    for thread in threads:
        thread.start()

    seen_paths: set[str] = set()
    invalid_by_path: OrderedDict[str, str] = OrderedDict()
    done_workers = 0
    pending_error: BaseException | None = None
    try:
        while done_workers < len(threads):
            item = output_queue.get()
            if item is _DONE:
                done_workers += 1
                continue
            if isinstance(item, _WorkerError):
                stop_event.set()
                pending_error = item.error
                continue
            if isinstance(item, _InvalidVideo):
                if item.video_path not in seen_paths:
                    invalid_by_path.setdefault(item.video_path, item.error)
                continue
            rows = item
            if not isinstance(rows, list):
                raise TypeError(f"Unexpected Kinetics-700 worker output: {type(rows).__name__}")
            if pending_error is not None:
                continue
            deduplicated_rows: list[dict[str, Any]] = []
            for row in rows:
                video_path = row["frame_unit_key"]
                if video_path in seen_paths:
                    continue
                seen_paths.add(video_path)
                invalid_by_path.pop(video_path, None)
                valid_keys.add(video_path)
                deduplicated_rows.append(row)
            if deduplicated_rows:
                yield pa.RecordBatch.from_pylist(deduplicated_rows, schema=FRAME_UNIT_SCHEMA)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    invalid_video_rows.extend(
        {"video_path": video_path, "error": error}
        for video_path, error in invalid_by_path.items()
    )
    missing_paths = [path for path in video_refs if path not in seen_paths and path not in invalid_by_path]
    if pending_error is not None:
        raise pending_error
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        raise FileNotFoundError(
            f"Kinetics-700 train tars are missing {len(missing_paths)} referenced videos. "
            f"First missing paths: {preview}"
        )


def _frame_unit_batches_from_tar_paths(
    *,
    tar_paths: list[Path],
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
    invalid_video_policy: str,
    invalid_video_rows: list[dict[str, str]],
    full_video_max_frames: int,
    progress_label: str | None = None,
    shard_index: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Build frame-unit batches from a fixed tar-path subset."""

    if invalid_video_policy not in {"error", "skip"}:
        raise ValueError("invalid_video_policy must be 'error' or 'skip'.")
    if not video_refs or not tar_paths:
        return

    workers = max(1, min(int(num_workers), len(tar_paths)))
    path_groups = [tar_paths[index::workers] for index in range(workers)]
    video_id_by_path = dict(video_refs)
    output_queue: queue.Queue[object] = queue.Queue(maxsize=max(2, workers * 2))
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_scan_tar_frame_unit_worker,
            kwargs={
                "paths": group,
                "video_id_by_path": video_id_by_path,
                "output_queue": output_queue,
                "batch_size": batch_size,
                "stop_event": stop_event,
                "invalid_video_policy": invalid_video_policy,
                "full_video_max_frames": full_video_max_frames,
                "progress_label": progress_label,
                "shard_index": shard_index,
            },
            daemon=True,
        )
        for group in path_groups
        if group
    ]
    for thread in threads:
        thread.start()

    invalid_by_path: OrderedDict[str, str] = OrderedDict()
    done_workers = 0
    pending_error: BaseException | None = None
    try:
        while done_workers < len(threads):
            item = output_queue.get()
            if item is _DONE:
                done_workers += 1
                continue
            if isinstance(item, _WorkerError):
                stop_event.set()
                pending_error = item.error
                continue
            if isinstance(item, _InvalidVideo):
                invalid_by_path.setdefault(item.video_path, item.error)
                continue
            rows = item
            if not isinstance(rows, list):
                raise TypeError(f"Unexpected Kinetics-700 worker output: {type(rows).__name__}")
            if pending_error is not None:
                continue
            yield pa.RecordBatch.from_pylist(rows, schema=FRAME_UNIT_SCHEMA)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    invalid_video_rows.extend(
        {"video_path": video_path, "error": error}
        for video_path, error in invalid_by_path.items()
    )
    if pending_error is not None:
        raise pending_error


def _video_batches_from_tar_paths(
    *,
    tar_paths: list[Path],
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
    invalid_video_policy: str,
    invalid_video_rows: list[dict[str, str]],
    progress_label: str | None = None,
    shard_index: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Build video table batches from a fixed tar-path subset."""

    if invalid_video_policy not in {"error", "skip"}:
        raise ValueError("invalid_video_policy must be 'error' or 'skip'.")
    if not video_refs or not tar_paths:
        return

    workers = max(1, min(int(num_workers), len(tar_paths)))
    path_groups = [tar_paths[index::workers] for index in range(workers)]
    video_id_by_path = dict(video_refs)
    output_queue: queue.Queue[object] = queue.Queue(maxsize=max(2, workers * 2))
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_scan_tar_worker,
            kwargs={
                "paths": group,
                "video_id_by_path": video_id_by_path,
                "output_queue": output_queue,
                "batch_size": batch_size,
                "stop_event": stop_event,
                "invalid_video_policy": invalid_video_policy,
                "progress_label": progress_label,
                "shard_index": shard_index,
            },
            daemon=True,
        )
        for group in path_groups
        if group
    ]
    for thread in threads:
        thread.start()

    invalid_by_path: OrderedDict[str, str] = OrderedDict()
    done_workers = 0
    pending_error: BaseException | None = None
    try:
        while done_workers < len(threads):
            item = output_queue.get()
            if item is _DONE:
                done_workers += 1
                continue
            if isinstance(item, _WorkerError):
                stop_event.set()
                pending_error = item.error
                continue
            if isinstance(item, _InvalidVideo):
                invalid_by_path.setdefault(item.video_path, item.error)
                continue
            rows = item
            if not isinstance(rows, list):
                raise TypeError(f"Unexpected Kinetics-700 worker output: {type(rows).__name__}")
            if pending_error is not None:
                continue
            yield pa.RecordBatch.from_pylist(rows, schema=VIDEOS_SCHEMA)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    invalid_video_rows.extend(
        {"video_path": video_path, "error": error}
        for video_path, error in invalid_by_path.items()
    )
    if pending_error is not None:
        raise pending_error


def _bad_media_rows_from_invalid_rows(
    invalid_video_rows: list[dict[str, str]],
    video_refs: OrderedDict[str, str],
) -> list[dict[str, Any]]:
    """Convert legacy invalid-video rows to the common bad-media schema."""

    return [
        build_bad_media_row(
            dataset="Kinetics-700",
            split="all",
            source_key=video_refs.get(row["video_path"], ""),
            frame_unit_key=row["video_path"],
            reason="invalid_video",
            error=row["error"],
            source_path=row["video_path"],
        )
        for row in invalid_video_rows
    ]


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


def _scan_lance_batches(
    dataset_path: Path,
    *,
    schema: pa.Schema,
    key_column: str,
    seen_keys: set[str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield deduplicated batches from an existing Lance table."""

    dataset = lance.dataset(str(dataset_path))
    key_index = schema.names.index(key_column)
    for batch in dataset.scanner(batch_size=batch_size, batch_readahead=2).to_batches():
        keys = batch.column(key_index).to_pylist()
        keep_indices: list[int] = []
        for index, key in enumerate(keys):
            normalized = str(key)
            if normalized in seen_keys:
                continue
            seen_keys.add(normalized)
            keep_indices.append(index)
        if not keep_indices:
            continue
        if len(keep_indices) == batch.num_rows:
            yield batch
        else:
            yield batch.take(pa.array(keep_indices, type=pa.int64()))


def _merge_lance_shards(
    *,
    shard_paths: list[Path],
    table_name: str,
    output_path: Path,
    schema: pa.Schema,
    key_column: str,
    batch_size: int,
    max_bytes_per_file: int,
) -> tuple[int, set[str]]:
    """Merge per-shard Lance tables into one deduplicated table."""

    existing_paths = [path / "data" / table_name for path in shard_paths if (path / "data" / table_name).exists()]
    seen_keys: set[str] = set()

    def batches() -> Iterator[pa.RecordBatch]:
        for table_path in existing_paths:
            yield from _scan_lance_batches(
                table_path,
                schema=schema,
                key_column=key_column,
                seen_keys=seen_keys,
                batch_size=batch_size,
            )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, batches()),
        str(output_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    return len(seen_keys), seen_keys


def _write_kinetics700_shard(
    *,
    shard_index: int,
    tar_paths: list[Path],
    video_refs: OrderedDict[str, str],
    output_root: Path,
    video_batch_size: int,
    threads_per_shard: int,
    max_bytes_per_file: int,
    invalid_video_policy: str,
    full_video_max_frames: int,
) -> dict[str, Any]:
    """Write one independent Kinetics-700 shard root."""

    shard_root = output_root / "_shards" / f"shard-{shard_index:05d}"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    (shard_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    invalid_video_rows: list[dict[str, str]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches_from_tar_paths(
                tar_paths=tar_paths,
                video_refs=video_refs,
                batch_size=video_batch_size,
                num_workers=threads_per_shard,
                invalid_video_policy=invalid_video_policy,
                invalid_video_rows=invalid_video_rows,
                full_video_max_frames=full_video_max_frames,
                progress_label="frames",
                shard_index=shard_index,
            ),
        ),
        str(shard_root / "data" / "frames.lance"),
        schema=FRAME_UNIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    valid_keys: set[str] = set()
    frames_dataset = lance.dataset(str(shard_root / "data" / "frames.lance"))
    key_index = FRAME_UNIT_SCHEMA.names.index("frame_unit_key")
    for batch in frames_dataset.scanner(columns=["frame_unit_key"], batch_size=8192).to_batches():
        valid_keys.update(str(value) for value in batch.column(key_index).to_pylist())
    valid_video_refs = OrderedDict(
        (video_path, video_id)
        for video_path, video_id in video_refs.items()
        if video_path in valid_keys
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches_from_tar_paths(
                tar_paths=tar_paths,
                video_refs=valid_video_refs,
                batch_size=video_batch_size,
                num_workers=threads_per_shard,
                invalid_video_policy=invalid_video_policy,
                invalid_video_rows=[],
                progress_label="videos",
                shard_index=shard_index,
            ),
        ),
        str(shard_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    _write_rows(
        shard_root / "data" / "exclusions" / "invalid_videos.lance",
        invalid_video_rows,
        schema=INVALID_VIDEO_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )
    return {
        "shard_index": shard_index,
        "shard_root": str(shard_root),
        "tar_count": len(tar_paths),
        "frame_unit_rows": len(valid_keys),
        "invalid_video_rows": len(invalid_video_rows),
        "invalid_video_examples": invalid_video_rows[:10],
    }


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_kinetics700_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Kinetics-700 runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing Kinetics-700 frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_path")


def write_kinetics700_root(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    invalid_video_policy: str = "error",
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
) -> dict[str, Any]:
    """Convert Kinetics-700 into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    exclusions = load_mmeb_v2_eval_exclusion_rows(vlm2vec_root)
    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    invalid_video_rows: list[dict[str, str]] = []
    valid_video_keys: set[str] = set()
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches_from_tars(
                raw_root=raw_root,
                video_refs=video_refs,
                batch_size=video_batch_size,
                num_workers=num_workers,
                invalid_video_policy=invalid_video_policy,
                invalid_video_rows=invalid_video_rows,
                valid_keys=valid_video_keys,
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
    video_invalid_rows: list[dict[str, str]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches_from_tars(
                raw_root=raw_root,
                video_refs=valid_video_refs,
                batch_size=video_batch_size,
                num_workers=num_workers,
                invalid_video_policy=invalid_video_policy,
                invalid_video_rows=video_invalid_rows,
            ),
        ),
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    invalid_paths = set(video_refs) - valid_video_keys
    split_rows = load_kinetics700_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    final_split_counts: dict[str, int] = {}
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video_path"] not in invalid_paths]
        final_split_counts[split_name] = len(filtered_rows)
        reader = pa.RecordBatchReader.from_batches(
            TRAIN_SPLIT_SCHEMA,
            _split_batches(filtered_rows, schema=TRAIN_SPLIT_SCHEMA, batch_size=split_batch_size),
        )
        lance.write_dataset(
            reader,
            str(output_root / "data" / f"{split_name}.lance"),
            schema=TRAIN_SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    _write_rows(
        output_root / "data" / "exclusions" / "mmeb_v2_eval.lance",
        exclusions,
        schema=EXCLUSION_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )
    _write_rows(
        output_root / "data" / "exclusions" / "invalid_videos.lance",
        invalid_video_rows,
        schema=INVALID_VIDEO_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )
    _write_rows(
        output_root / "data" / "exclusions" / "bad_media.lance",
        _bad_media_rows_from_invalid_rows(invalid_video_rows, video_refs),
        schema=BAD_MEDIA_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_kinetics700_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "excluded_rows": final_split_counts["official_train"] - final_split_counts["train_without_mmeb_v2_eval"],
        "exclusion_rows": len(exclusions),
        "invalid_video_rows": len(invalid_video_rows),
        "bad_media_rows": len(invalid_video_rows),
        "invalid_video_examples": invalid_video_rows[:10],
        "splits": final_split_counts,
        "raw_splits": split_counts,
        "artifact_format": "kinetics700_train_frame_unit_lance_v1",
    }


def write_kinetics700_root_sharded(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
    output_root: Path,
    overwrite: bool = False,
    shard_workers: int = 8,
    threads_per_shard: int = 1,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    merge_batch_size: int = 128,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    invalid_video_policy: str = "error",
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    keep_shards: bool = False,
) -> dict[str, Any]:
    """Convert Kinetics-700 with process-level tar sharding."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    started_at = time.monotonic()
    exclusions = load_mmeb_v2_eval_exclusion_rows(vlm2vec_root)
    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    tar_paths = _tar_paths(raw_root)
    workers = max(1, min(int(shard_workers), len(tar_paths)))
    path_groups = [tar_paths[index::workers] for index in range(workers)]
    shard_roots = [output_root / "_shards" / f"shard-{index:05d}" for index in range(workers)]
    _emit_progress(
        "kinetics700_sharded_start",
        output_root=str(output_root),
        tar_count=len(tar_paths),
        shard_workers=workers,
        threads_per_shard=threads_per_shard,
        video_refs=len(video_refs),
    )

    shard_summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [
            executor.submit(
                _write_kinetics700_shard,
                shard_index=index,
                tar_paths=group,
                video_refs=video_refs,
                output_root=output_root,
                video_batch_size=video_batch_size,
                threads_per_shard=threads_per_shard,
                max_bytes_per_file=max_bytes_per_file,
                invalid_video_policy=invalid_video_policy,
                full_video_max_frames=full_video_max_frames,
            )
            for index, group in enumerate(path_groups)
            if group
        ]
        for future in as_completed(futures):
            summary = future.result()
            shard_summaries.append(summary)
            _emit_progress("kinetics700_shard_done", **summary)

    shard_summaries.sort(key=lambda item: int(item["shard_index"]))
    invalid_video_rows: list[dict[str, str]] = []
    for shard_root in shard_roots:
        invalid_path = shard_root / "data" / "exclusions" / "invalid_videos.lance"
        if not invalid_path.exists():
            continue
        dataset = lance.dataset(str(invalid_path))
        for batch in dataset.scanner(batch_size=8192).to_batches():
            invalid_video_rows.extend(batch.to_pylist())

    _emit_progress("kinetics700_merge_start", table="frames.lance", shards=len(shard_roots))
    frame_unit_rows, valid_video_keys = _merge_lance_shards(
        shard_paths=shard_roots,
        table_name="frames.lance",
        output_path=output_root / "data" / "frames.lance",
        schema=FRAME_UNIT_SCHEMA,
        key_column="frame_unit_key",
        batch_size=merge_batch_size,
        max_bytes_per_file=max_bytes_per_file,
    )
    _emit_progress("kinetics700_merge_done", table="frames.lance", rows=frame_unit_rows)

    invalid_paths = {row["video_path"] for row in invalid_video_rows}
    missing_paths = [
        path
        for path in video_refs
        if path not in valid_video_keys and path not in invalid_paths
    ]
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        raise FileNotFoundError(
            f"Kinetics-700 train tars are missing {len(missing_paths)} referenced videos. "
            f"First missing paths: {preview}"
        )

    _emit_progress("kinetics700_merge_start", table="videos.lance", shards=len(shard_roots))
    video_rows, _ = _merge_lance_shards(
        shard_paths=shard_roots,
        table_name="videos.lance",
        output_path=output_root / "data" / "videos.lance",
        schema=VIDEOS_SCHEMA,
        key_column="video_path",
        batch_size=merge_batch_size,
        max_bytes_per_file=max_bytes_per_file,
    )
    _emit_progress("kinetics700_merge_done", table="videos.lance", rows=video_rows)

    invalid_or_missing_paths = set(video_refs) - valid_video_keys
    split_rows = load_kinetics700_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    final_split_counts: dict[str, int] = {}
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video_path"] not in invalid_or_missing_paths]
        final_split_counts[split_name] = len(filtered_rows)
        reader = pa.RecordBatchReader.from_batches(
            TRAIN_SPLIT_SCHEMA,
            _split_batches(filtered_rows, schema=TRAIN_SPLIT_SCHEMA, batch_size=split_batch_size),
        )
        lance.write_dataset(
            reader,
            str(output_root / "data" / f"{split_name}.lance"),
            schema=TRAIN_SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    _write_rows(
        output_root / "data" / "exclusions" / "mmeb_v2_eval.lance",
        exclusions,
        schema=EXCLUSION_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )
    _write_rows(
        output_root / "data" / "exclusions" / "invalid_videos.lance",
        invalid_video_rows,
        schema=INVALID_VIDEO_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )
    _write_rows(
        output_root / "data" / "exclusions" / "bad_media.lance",
        _bad_media_rows_from_invalid_rows(invalid_video_rows, video_refs),
        schema=BAD_MEDIA_SCHEMA,
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_kinetics700_indices(output_root)
    if not keep_shards:
        shutil.rmtree(output_root / "_shards", ignore_errors=True)
    summary = {
        "output_root": str(output_root),
        "video_rows": video_rows,
        "frame_unit_rows": frame_unit_rows,
        "excluded_rows": final_split_counts["official_train"] - final_split_counts["train_without_mmeb_v2_eval"],
        "exclusion_rows": len(exclusions),
        "invalid_video_rows": len(invalid_video_rows),
        "bad_media_rows": len(invalid_video_rows),
        "invalid_video_examples": invalid_video_rows[:10],
        "splits": final_split_counts,
        "raw_splits": split_counts,
        "shards": shard_summaries,
        "shard_workers": workers,
        "threads_per_shard": threads_per_shard,
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "artifact_format": "kinetics700_train_frame_unit_lance_v1",
    }
    _emit_progress("kinetics700_sharded_done", **summary)
    return summary


__all__ = [
    "EXCLUSION_SCHEMA",
    "INVALID_VIDEO_SCHEMA",
    "SUPPORTED_SPLITS",
    "TRAIN_SPLIT_SCHEMA",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_kinetics700_indices",
    "load_kinetics700_split_rows",
    "load_mmeb_v2_eval_exclusion_rows",
    "write_kinetics700_root",
    "write_kinetics700_root_sharded",
]
