"""NExTQA training conversion utilities."""

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
    FULL_VIDEO_MAX_FRAMES,
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = ("official_train", "official_train_without_mmeb_v2_eval")
OPTION_LIST_TYPE = pa.list_(pa.string())
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video", pa.string()),
        pa.field("frame_count", pa.string()),
        pa.field("width", pa.string()),
        pa.field("height", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("qid", pa.string()),
        pa.field("type", pa.string()),
        pa.field("options", OPTION_LIST_TYPE),
        pa.field("video_path", pa.string()),
    ]
)
VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video_path", pa.string()),
        pa.field("video", pa.string()),
        pa.field("member_path", pa.string()),
        pa.field("video_bytes", pa.binary()),
        pa.field("fps", pa.float32()),
        pa.field("total_num_frames", pa.int32()),
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


def _load_json_rows(annotation_path: Path) -> list[dict[str, Any]]:
    """Load NExTQA JSON annotation rows."""

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing NExTQA annotation file: {annotation_path}")
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"NExTQA annotation must contain a JSON list: {annotation_path}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"NExTQA row {index} must be an object.")
        normalized.append(dict(row))
    return normalized


def _normalize_options(row: dict[str, Any]) -> list[str]:
    """Return the five NExTQA multiple-choice options."""

    options = row.get("options")
    if not isinstance(options, list) or len(options) != 5:
        raise ValueError(
            f"NExTQA row must contain exactly five options: video={row.get('video')!r}, qid={row.get('qid')!r}"
        )
    return [str(option) for option in options]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw row while preserving source fields."""

    options = _normalize_options(row)
    return {
        "video": str(row["video"]),
        "frame_count": str(row.get("frame_count", "") or ""),
        "width": str(row.get("width", "") or ""),
        "height": str(row.get("height", "") or ""),
        "question": str(row.get("question", "") or ""),
        "answer": str(row["answer"]),
        "qid": str(row.get("qid", "") or ""),
        "type": str(row.get("type", "") or ""),
        "options": options,
        "video_path": str(row["video_path"]),
    }


def load_nextqa_train_rows(annotation_path: Path) -> list[dict[str, Any]]:
    """Return source-preserving NExTQA train rows."""

    return [_normalize_row(row) for row in _load_json_rows(annotation_path)]


def load_nextqa_split_rows(annotation_path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Return NExTQA official train and eval-excluded train rows."""

    rows = load_nextqa_train_rows(annotation_path)
    return OrderedDict(
        [
            ("official_train", rows),
            ("official_train_without_mmeb_v2_eval", list(rows)),
        ]
    )


def collect_video_refs(annotation_path: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading videos."""

    video_by_path: OrderedDict[str, str] = OrderedDict()
    split_rows = load_nextqa_split_rows(annotation_path)
    counts: dict[str, int] = {}
    for split_name, rows in split_rows.items():
        count = 0
        for row in rows:
            video_by_path.setdefault(str(row["video_path"]), str(row["video"]))
            count += 1
        counts[split_name] = count
    return video_by_path, counts


def _zip_member_index(video_zip: Path) -> dict[str, str]:
    """Map source ``video_path`` values to zip members."""

    if not video_zip.is_file():
        raise FileNotFoundError(f"Missing NExTQA video zip: {video_zip}")
    members: dict[str, str] = {}
    with zipfile.ZipFile(video_zip) as archive:
        for name in archive.namelist():
            if not name.endswith(".mp4"):
                continue
            normalized = name.removeprefix("NExTVideo/").removesuffix(".mp4")
            members[normalized] = name
    if not members:
        raise FileNotFoundError(f"No mp4 videos found in NExTQA zip: {video_zip}")
    return members


def _zip_reader(zip_path: Path, member_path: str) -> bytes:
    """Read one zip member and close the archive handle immediately."""

    try:
        with zipfile.ZipFile(zip_path) as handle:
            return handle.read(member_path)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing NExTQA video member '{member_path}' in archive {zip_path}") from exc


def _build_video_row(video_zip: Path, item: tuple[str, str, str]) -> dict[str, Any]:
    """Build one Lance-ready video row from a zip member."""

    video_path, video, member_path = item
    video_bytes = _zip_reader(video_zip, member_path)
    metadata = probe_video_bytes(video_bytes, source_path=member_path)
    return {
        "video_path": video_path,
        "video": video,
        "member_path": member_path,
        "video_bytes": video_bytes,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _video_batches(
    *,
    video_zip: Path,
    video_refs: OrderedDict[str, str],
    member_index: dict[str, str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build ordered video batches from the NExTQA zip archive."""

    missing = sorted(set(video_refs) - set(member_index))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing NExTQA videos in zip archive: {preview}")

    items = [
        (video_path, video, member_index[video_path])
        for video_path, video in video_refs.items()
    ]
    workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = executor.map(lambda item: _build_video_row(video_zip, item), items)
        for batch in _batched(rows, batch_size):
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)


def _build_frame_unit_row(video_zip: Path, item: tuple[str, str, str], *, max_frames: int) -> dict[str, Any]:
    """Build one full-video frame-unit from a zip member."""

    video_path, video, member_path = item
    return build_full_video_frame_unit_from_bytes(
        video_bytes=_zip_reader(video_zip, member_path),
        frame_unit_key=video_path,
        source_key=video,
        source_path=member_path,
        max_frames=max_frames,
    )


def _frame_unit_batches(
    *,
    video_zip: Path,
    video_refs: OrderedDict[str, str],
    member_index: dict[str, str],
    batch_size: int,
    num_workers: int,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build frame-unit batches from the NExTQA zip archive."""

    missing = sorted(set(video_refs) - set(member_index))
    for video_path in missing:
        bad_media_rows.append(
            build_bad_media_row(
                dataset="NExTQA",
                split="all",
                source_key=video_refs[video_path],
                frame_unit_key=video_path,
                reason="missing_video",
                error=f"Missing NExTQA video in zip archive: {video_path}",
                source_path=video_path,
            )
        )

    items = [
        (video_path, video, member_index[video_path])
        for video_path, video in video_refs.items()
        if video_path in member_index
    ]
    workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            (item, executor.submit(_build_frame_unit_row, video_zip, item, max_frames=full_video_max_frames))
            for item in items
        ]
        rows: list[dict[str, Any]] = []
        for item, future in futures:
            video_path, video, member_path = item
            try:
                row = future.result()
            except Exception as error:
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="NExTQA",
                        split="all",
                        source_key=video,
                        frame_unit_key=video_path,
                        reason="invalid_video",
                        error=str(error),
                        source_path=member_path,
                    )
                )
                continue
            rows.append(row)
            valid_keys.add(video_path)
            if len(rows) >= batch_size:
                yield pa.RecordBatch.from_pylist(rows, schema=FRAME_UNIT_SCHEMA)
                rows = []
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


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SPLIT_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_nextqa_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the NExTQA runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing NExTQA frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_path")


def write_nextqa_root(
    *,
    annotation_path: Path,
    video_zip: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 32,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert NExTQA into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    video_refs, split_counts = collect_video_refs(annotation_path)
    member_index = _zip_member_index(video_zip)
    valid_video_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                video_zip=video_zip,
                video_refs=video_refs,
                member_index=member_index,
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
        (video_path, video)
        for video_path, video in video_refs.items()
        if video_path in valid_video_keys
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches(
                video_zip=video_zip,
                video_refs=valid_video_refs,
                member_index=member_index,
                batch_size=video_batch_size,
                num_workers=num_workers,
            ),
        ),
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_nextqa_split_rows(annotation_path)
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video_path"] in valid_video_keys]
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

    empty_exclusion_reader = pa.RecordBatchReader.from_batches(
        SPLIT_SCHEMA,
        _split_batches([], batch_size=split_batch_size),
    )
    lance.write_dataset(
        empty_exclusion_reader,
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

    ensure_nextqa_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "splits": split_counts,
        "exclusion_rows": 0,
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "nextqa_train_frame_unit_lance_v1",
    }


__all__ = [
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_nextqa_indices",
    "load_nextqa_split_rows",
    "load_nextqa_train_rows",
    "write_nextqa_root",
]
