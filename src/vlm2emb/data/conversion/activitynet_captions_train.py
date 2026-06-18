"""ActivityNet Captions training conversion utilities."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE
from vlm2emb.data.conversion.video_frame_units import (
    BAD_MEDIA_SCHEMA,
    FRAME_UNIT_SCHEMA,
    SEGMENT_MAX_FRAMES,
    build_bad_media_row,
    build_segment_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = (
    "official_train",
    "official_val1",
    "official_val2",
    "official_train_without_mmeb_v2_eval",
)
TIMESTAMP_LIST_TYPE = pa.list_(pa.list_(pa.float64()))
SENTENCE_LIST_TYPE = pa.list_(pa.string())
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("source", pa.string()),
        pa.field("duration", pa.float64()),
        pa.field("timestamps", TIMESTAMP_LIST_TYPE),
        pa.field("sentences", SENTENCE_LIST_TYPE),
        pa.field("segment_index", pa.int32()),
        pa.field("segment_start", pa.float64()),
        pa.field("segment_end", pa.float64()),
        pa.field("segment_sentence", pa.string()),
        pa.field("frame_unit_key", pa.string()),
    ]
)
VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video", pa.string()),
        pa.field("video_id", pa.string()),
        pa.field("member_path", pa.string()),
        pa.field("video_bytes", pa.binary()),
        pa.field("fps", pa.float32()),
        pa.field("total_num_frames", pa.int32()),
    ]
)
RAW_SPLIT_FILES = OrderedDict(
    [
        ("official_train", "activitynet_captions_train.json"),
        ("official_val1", "activitynet_captions_val1.json"),
        ("official_val2", "activitynet_captions_val2.json"),
    ]
)


class ConcatenatedFile(io.RawIOBase):
    """Read multiple file parts as one sequential binary stream."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = [Path(path) for path in paths]
        self._index = 0
        self._handle: io.BufferedReader | None = None

    def readable(self) -> bool:
        """Return whether this stream can be read."""

        return True

    def _ensure_handle(self) -> io.BufferedReader | None:
        while self._handle is None and self._index < len(self._paths):
            self._handle = self._paths[self._index].open("rb")
            self._index += 1
        return self._handle

    def read(self, size: int = -1) -> bytes:
        """Read bytes across part boundaries."""

        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while size < 0 or remaining > 0:
            handle = self._ensure_handle()
            if handle is None:
                break
            chunk = handle.read(-1 if size < 0 else remaining)
            if chunk:
                chunks.append(chunk)
                if size > 0:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
                continue
            handle.close()
            self._handle = None
        return b"".join(chunks)

    def close(self) -> None:
        """Close the active part handle."""

        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


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
    """Load one ActivityNet Captions JSON split file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing ActivityNet Captions split file: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"ActivityNet Captions split file must contain a JSON list: {path}")
    return [dict(row) for row in rows]


def _normalize_timestamps(value: Any, *, video_id: str) -> list[list[float]]:
    """Normalize ActivityNet timestamp pairs."""

    if not isinstance(value, list):
        raise TypeError(f"ActivityNet Captions timestamps must be a list: video_id={video_id!r}")
    timestamps: list[list[float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"ActivityNet Captions timestamp must be [start, end]: video_id={video_id!r}")
        timestamps.append([float(item[0]), float(item[1])])
    return timestamps


def _normalize_video_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one source video-level annotation row."""

    video_id = str(row["video_id"])
    sentences = row.get("sentences")
    if not isinstance(sentences, list):
        raise TypeError(f"ActivityNet Captions sentences must be a list: video_id={video_id!r}")
    timestamps = _normalize_timestamps(row.get("timestamps"), video_id=video_id)
    if len(sentences) != len(timestamps):
        raise ValueError(
            f"ActivityNet Captions sentences/timestamps length mismatch: video_id={video_id!r}"
        )
    return {
        "video_id": video_id,
        "video": str(row["video"]),
        "caption": str(row.get("caption", "") or ""),
        "source": str(row.get("source", "ActivityNet_Captions") or "ActivityNet_Captions"),
        "duration": float(row.get("duration", 0.0) or 0.0),
        "timestamps": timestamps,
        "sentences": [str(sentence) for sentence in sentences],
    }


def _expand_segments(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand one video-level row into segment-level training rows."""

    for index, (timestamp, sentence) in enumerate(zip(row["timestamps"], row["sentences"], strict=True)):
        start_time = float(timestamp[0])
        end_time = float(timestamp[1])
        yield {
            **row,
            "segment_index": index,
            "segment_start": start_time,
            "segment_end": end_time,
            "segment_sentence": str(sentence),
            "frame_unit_key": _segment_frame_unit_key(str(row["video"]), index, start_time, end_time),
        }


def load_activitynet_captions_split_rows(source_root: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Return ActivityNet Captions segment-level split rows."""

    rows_by_split: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for split_name, filename in RAW_SPLIT_FILES.items():
        video_rows = [_normalize_video_row(row) for row in _load_json_rows(source_root / filename)]
        rows_by_split[split_name] = [
            segment
            for row in video_rows
            for segment in _expand_segments(row)
        ]
    rows_by_split["official_train_without_mmeb_v2_eval"] = list(rows_by_split["official_train"])
    return rows_by_split


def collect_video_refs(source_root: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique videos and split row counts without reading video bytes."""

    video_id_by_file: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_activitynet_captions_split_rows(source_root).items():
        count = 0
        for row in rows:
            video_id_by_file.setdefault(str(row["video"]), str(row["video_id"]))
            count += 1
        counts[split_name] = count
    return video_id_by_file, counts


def _open_tar_stream(paths: Sequence[Path]) -> tarfile.TarFile:
    """Open one tar or split-tar sequence for streaming reads."""

    if not paths:
        raise ValueError("At least one ActivityNet video archive path is required.")
    normalized = [Path(path) for path in paths]
    if len(normalized) == 1:
        return tarfile.open(normalized[0], mode="r:*")
    return tarfile.open(fileobj=ConcatenatedFile(normalized), mode="r|*")


def _build_video_row(video: str, video_id: str, member_path: str, video_bytes: bytes) -> dict[str, Any]:
    """Build one Lance-ready ActivityNet video row."""

    metadata = probe_video_bytes(video_bytes, source_path=member_path)
    return {
        "video": video,
        "video_id": video_id,
        "member_path": member_path,
        "video_bytes": video_bytes,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _segment_frame_unit_key(video: str, segment_index: int, start_time: float, end_time: float) -> str:
    """Build a stable key for one ActivityNet temporal segment."""

    return f"{video}#segment={int(segment_index)}:{float(start_time):.3f}-{float(end_time):.3f}"


def _segments_by_video(source_root: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Collect unique ActivityNet segment rows by source video."""

    grouped: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
    for rows in load_activitynet_captions_split_rows(source_root).values():
        for row in rows:
            video = str(row["video"])
            grouped.setdefault(video, OrderedDict())
            grouped[video].setdefault(str(row["frame_unit_key"]), row)
    return OrderedDict((video, list(rows.values())) for video, rows in grouped.items())


def _iter_video_rows_from_archives(
    *,
    archive_paths: Sequence[Path],
    video_refs: OrderedDict[str, str],
) -> Iterator[dict[str, Any]]:
    """Yield selected video rows from split tar archives."""

    member_to_video = {Path(video).name: video for video in video_refs}
    found: set[str] = set()
    with _open_tar_stream(archive_paths) as archive:
        for member in archive:
            member_filename = Path(member.name).name
            if not member.isfile() or member_filename not in member_to_video:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read ActivityNet tar member: {member.name}")
            video = member_to_video[member_filename]
            found.add(video)
            yield _build_video_row(video, video_refs[video], member.name, handle.read())

    missing = sorted(set(video_refs) - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing ActivityNet videos in tar archives: {preview}")


def _video_batches(
    *,
    archive_paths: Sequence[Path],
    video_refs: OrderedDict[str, str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Build video batches by streaming split tar archives once."""

    batch: list[dict[str, Any]] = []
    for row in _iter_video_rows_from_archives(archive_paths=archive_paths, video_refs=video_refs):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)


def _iter_frame_unit_rows_from_archives(
    *,
    archive_paths: Sequence[Path],
    video_refs: OrderedDict[str, str],
    segments_by_video: OrderedDict[str, list[dict[str, Any]]],
    valid_keys: set[str],
    valid_videos: set[str],
    bad_media_rows: list[dict[str, Any]],
    segment_max_frames: int,
) -> Iterator[dict[str, Any]]:
    """Yield segment frame-units from ActivityNet archives while collecting failures."""

    member_to_video = {Path(video).name: video for video in video_refs}
    found: set[str] = set()
    with _open_tar_stream(archive_paths) as archive:
        for member in archive:
            member_filename = Path(member.name).name
            if not member.isfile() or member_filename not in member_to_video:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read ActivityNet tar member: {member.name}")
            video = member_to_video[member_filename]
            found.add(video)
            video_bytes = handle.read()
            video_has_valid_segment = False
            for segment in segments_by_video.get(video, []):
                frame_unit_key = str(segment["frame_unit_key"])
                try:
                    row = build_segment_frame_unit_from_bytes(
                        video_bytes=video_bytes,
                        frame_unit_key=frame_unit_key,
                        source_key=str(segment["video_id"]),
                        source_path=member.name,
                        start_time=float(segment["segment_start"]),
                        end_time=float(segment["segment_end"]),
                        max_frames=segment_max_frames,
                    )
                except Exception as error:
                    bad_media_rows.append(
                        build_bad_media_row(
                            dataset="ActivityNet-Captions",
                            split="all",
                            source_key=str(segment["video_id"]),
                            frame_unit_key=frame_unit_key,
                            reason="invalid_segment",
                            error=str(error),
                            source_path=member.name,
                        )
                    )
                    continue
                valid_keys.add(frame_unit_key)
                video_has_valid_segment = True
                yield row
            if video_has_valid_segment:
                valid_videos.add(video)

    missing = sorted(set(video_refs) - found)
    for missing_video in missing:
        bad_media_rows.append(
            build_bad_media_row(
                dataset="ActivityNet-Captions",
                split="all",
                source_key=video_refs[missing_video],
                frame_unit_key=missing_video,
                reason="missing_video",
                error=f"Missing ActivityNet video in tar archives: {missing_video}",
                source_path=missing_video,
            )
        )


def _frame_unit_batches(
    *,
    archive_paths: Sequence[Path],
    video_refs: OrderedDict[str, str],
    segments_by_video: OrderedDict[str, list[dict[str, Any]]],
    batch_size: int,
    valid_keys: set[str],
    valid_videos: set[str],
    bad_media_rows: list[dict[str, Any]],
    segment_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build segment frame-unit batches by streaming split tar archives once."""

    batch: list[dict[str, Any]] = []
    for row in _iter_frame_unit_rows_from_archives(
        archive_paths=archive_paths,
        video_refs=video_refs,
        segments_by_video=segments_by_video,
        valid_keys=valid_keys,
        valid_videos=valid_videos,
        bad_media_rows=bad_media_rows,
        segment_max_frames=segment_max_frames,
    ):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=FRAME_UNIT_SCHEMA)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=FRAME_UNIT_SCHEMA)


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


def ensure_activitynet_captions_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the ActivityNet Captions runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing ActivityNet Captions frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video")


def write_activitynet_captions_root(
    *,
    source_root: Path,
    video_archives: Sequence[Path],
    output_root: Path,
    overwrite: bool = False,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    segment_max_frames: int = SEGMENT_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert ActivityNet Captions into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    video_refs, split_counts = collect_video_refs(source_root)
    segments_by_video = _segments_by_video(source_root)
    valid_frame_unit_keys: set[str] = set()
    valid_videos: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                archive_paths=video_archives,
                video_refs=video_refs,
                segments_by_video=segments_by_video,
                batch_size=video_batch_size,
                valid_keys=valid_frame_unit_keys,
                valid_videos=valid_videos,
                bad_media_rows=bad_media_rows,
                segment_max_frames=segment_max_frames,
            )
        ),
        str(output_root / "data" / "frames.lance"),
        schema=FRAME_UNIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    valid_video_refs = OrderedDict(
        (video, video_id)
        for video, video_id in video_refs.items()
        if video in valid_videos
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches(
                archive_paths=video_archives,
                video_refs=valid_video_refs,
                batch_size=video_batch_size,
            ),
        ),
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_activitynet_captions_split_rows(source_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["frame_unit_key"] in valid_frame_unit_keys]
        split_counts[split_name] = len(filtered_rows)
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                SPLIT_SCHEMA,
                _split_batches(filtered_rows, batch_size=split_batch_size),
            ),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches([], batch_size=split_batch_size),
        ),
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

    ensure_activitynet_captions_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_frame_unit_keys),
        "splits": split_counts,
        "exclusion_rows": 0,
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "activitynet_captions_train_frame_unit_lance_v1",
    }


__all__ = [
    "RAW_SPLIT_FILES",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_activitynet_captions_indices",
    "load_activitynet_captions_split_rows",
    "write_activitynet_captions_root",
]
