"""QVHighlights training conversion utilities."""

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

SUPPORTED_SPLITS = ("official_train", "official_val", "official_test", "official_train_without_mmeb_v2_eval")

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
        pa.field("qid", pa.int64()),
        pa.field("query", pa.string()),
        pa.field("duration", pa.float32()),
        pa.field("vid", pa.string()),
        pa.field("relevant_clip_ids", pa.list_(pa.int64())),
        pa.field("saliency_scores", pa.list_(pa.list_(pa.int64()))),
        pa.field("relevant_windows", pa.list_(pa.list_(pa.float32()))),
        pa.field("query_frame_unit_key", pa.string()),
        pa.field("positive_frame_unit_keys", pa.list_(pa.string())),
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


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load one QVHighlights JSONL split while preserving source fields."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing QVHighlights split file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                row = {
                    "qid": int(obj["qid"]),
                    "query": str(obj["query"]),
                    "duration": float(obj["duration"]),
                    "vid": str(obj["vid"]),
                    "relevant_clip_ids": [int(value) for value in obj.get("relevant_clip_ids", [])],
                    "saliency_scores": [
                        [int(value) for value in values] for values in obj.get("saliency_scores", [])
                    ],
                    "relevant_windows": [
                        [float(window[0]), float(window[1])] for window in obj.get("relevant_windows", [])
                    ],
                }
                row["query_frame_unit_key"] = _query_frame_unit_key(str(row["vid"]))
                row["positive_frame_unit_keys"] = [
                    _segment_frame_unit_key(str(row["vid"]), float(window[0]), float(window[1]))
                    for window in row["relevant_windows"]
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid QVHighlights row {line_number} in {path}") from exc
            rows.append(row)
    return rows


def load_qvhighlight_split_rows(raw_root: Path) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return official train/val/test and eval-excluded train split rows."""

    train = _load_jsonl_rows(raw_root / "highlight_train_release.jsonl")
    val = _load_jsonl_rows(raw_root / "highlight_val_release.jsonl")
    test = _load_jsonl_rows(raw_root / "highlight_test_release.jsonl")
    eval_keys = {(row["qid"], row["vid"], row["query"]) for row in [*val, *test]}
    train_without_eval = [row for row in train if (row["qid"], row["vid"], row["query"]) not in eval_keys]
    return OrderedDict(
        [
            ("official_train", train),
            ("official_val", val),
            ("official_test", test),
            ("official_train_without_mmeb_v2_eval", train_without_eval),
        ]
    )


def collect_video_refs(raw_root: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique QVHighlights video refs and split row counts."""

    video_by_id: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_qvhighlight_split_rows(raw_root).items():
        count = 0
        for row in rows:
            video_id = str(row["vid"])
            video_by_id.setdefault(video_id, f"{video_id}.mp4")
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
    """Collect unique QVHighlights segment windows by source video id."""

    grouped: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
    for rows in load_qvhighlight_split_rows(raw_root).values():
        for row in rows:
            video_id = str(row["vid"])
            grouped.setdefault(video_id, OrderedDict())
            for window, frame_unit_key in zip(
                row["relevant_windows"],
                row["positive_frame_unit_keys"],
                strict=True,
            ):
                grouped[video_id].setdefault(
                    str(frame_unit_key),
                    {
                        "vid": video_id,
                        "start_time": float(window[0]),
                        "end_time": float(window[1]),
                        "frame_unit_key": str(frame_unit_key),
                    },
                )
    return OrderedDict((video_id, list(rows.values())) for video_id, rows in grouped.items())


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_qvhighlight_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the QVHighlights runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing QVHighlights frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_id")


def _build_video_row(video_root: Path, video_id: str, video: str) -> dict[str, Any]:
    """Build one Lance-ready QVHighlights video row."""

    video_path = video_root / video
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing QVHighlights video file: {video_path}")
    video_bytes = video_path.read_bytes()
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
    video_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build ordered video batches from local mp4 files."""

    items = list(video_refs.items())
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        rows = executor.map(lambda item: _build_video_row(video_root, item[0], item[1]), items)
        for batch in _batched(rows, batch_size):
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)


def _build_frame_unit_rows(
    video_root: Path,
    video_id: str,
    video: str,
    segments: list[dict[str, Any]],
    *,
    full_video_max_frames: int,
    segment_max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build mixed-view frame-units for one QVHighlights source video."""

    rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    video_path = video_root / video
    try:
        video_bytes = video_path.read_bytes()
    except Exception as error:
        bad_rows.append(
            build_bad_media_row(
                dataset="QVHighlights",
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
                dataset="QVHighlights",
                split="all",
                source_key=video_id,
                frame_unit_key=_query_frame_unit_key(video_id),
                reason="invalid_video",
                error=str(error),
                source_path=video,
            )
        )

    for segment in segments:
        frame_unit_key = str(segment["frame_unit_key"])
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
                    dataset="QVHighlights",
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
    video_root: Path,
    video_refs: OrderedDict[str, str],
    segments_by_video: OrderedDict[str, list[dict[str, Any]]],
    batch_size: int,
    num_workers: int,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
    segment_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build mixed-view frame-unit batches from local QVHighlights files."""

    items = list(video_refs.items())
    pending_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        results = executor.map(
            lambda item: _build_frame_unit_rows(
                video_root,
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


def _filter_trainable_row(row: dict[str, Any], valid_frame_unit_keys: set[str]) -> dict[str, Any] | None:
    """Return a row with only valid positive frame-units, or None if not trainable."""

    if row["query_frame_unit_key"] not in valid_frame_unit_keys:
        return None

    valid_windows: list[list[float]] = []
    valid_positive_keys: list[str] = []
    for window, frame_unit_key in zip(
        row["relevant_windows"],
        row["positive_frame_unit_keys"],
        strict=True,
    ):
        if frame_unit_key not in valid_frame_unit_keys:
            continue
        valid_windows.append(window)
        valid_positive_keys.append(frame_unit_key)
    if not valid_positive_keys:
        return None

    filtered = dict(row)
    filtered["relevant_windows"] = valid_windows
    filtered["positive_frame_unit_keys"] = valid_positive_keys
    return filtered


def write_qvhighlight_root(
    *,
    raw_root: Path,
    video_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    segment_max_frames: int = SEGMENT_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert QVHighlights into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    video_refs, split_counts = collect_video_refs(raw_root)
    segments_by_video = _segments_by_video(raw_root)
    valid_frame_unit_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                video_root=video_root,
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
                video_root=video_root,
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

    split_rows = load_qvhighlight_split_rows(raw_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [
            filtered_row
            for row in rows
            if (filtered_row := _filter_trainable_row(row, valid_frame_unit_keys)) is not None
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

    exclusion_rows = [*split_rows["official_val"], *split_rows["official_test"]]
    exclusion_reader = pa.RecordBatchReader.from_batches(
        SPLIT_SCHEMA,
        _split_batches(exclusion_rows, batch_size=split_batch_size),
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

    ensure_qvhighlight_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_frame_unit_keys),
        "splits": split_counts,
        "exclusion_rows": len(exclusion_rows),
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "qvhighlight_train_frame_unit_lance_v1",
    }


__all__ = [
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_qvhighlight_indices",
    "load_qvhighlight_split_rows",
    "write_qvhighlight_root",
]
