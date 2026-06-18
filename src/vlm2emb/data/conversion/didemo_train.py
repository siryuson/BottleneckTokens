"""DiDeMo training conversion utilities."""

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
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = ("train", "test", "train_without_mmeb_v2_eval")

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
        pa.field("video", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("source", pa.string()),
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
    """Load one DiDeMo JSON split file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing DiDeMo split file: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"DiDeMo split file must contain a JSON list: {path}")
    return [dict(row) for row in rows]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one DiDeMo row while preserving source field names."""

    caption = row.get("caption")
    if not isinstance(caption, str):
        raise TypeError(f"DiDeMo row caption must be a string: video={row.get('video')!r}")
    return {
        "video": str(row["video"]),
        "caption": caption,
        "source": str(row.get("source", "")),
    }


def load_didemo_split_rows(source_root: Path) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return DiDeMo train, test, and eval-excluded train split rows."""

    train = [_normalize_row(row) for row in _load_json_rows(source_root / "didemo_train.json")]
    test = [_normalize_row(row) for row in _load_json_rows(source_root / "didemo_test.json")]
    eval_videos = {row["video"] for row in test}
    train_without_eval = [row for row in train if row["video"] not in eval_videos]
    return OrderedDict(
        [
            ("train", train),
            ("test", test),
            ("train_without_mmeb_v2_eval", train_without_eval),
        ]
    )


def collect_video_refs(source_root: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading video bytes."""

    video_id_by_path: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_didemo_split_rows(source_root).items():
        count = 0
        for row in rows:
            video = str(row["video"])
            video_id_by_path.setdefault(video, Path(video).stem)
            count += 1
        counts[split_name] = count
    return video_id_by_path, counts


def _open_tar_stream(paths: Sequence[Path]) -> tarfile.TarFile:
    """Open one tar or split-tar sequence for streaming reads."""

    if not paths:
        raise ValueError("At least one tar path is required.")
    normalized = [Path(path) for path in paths]
    if len(normalized) == 1:
        return tarfile.open(normalized[0], mode="r:*")
    return tarfile.open(fileobj=ConcatenatedFile(normalized), mode="r|*")


def _build_video_row(video: str, video_bytes: bytes) -> dict[str, Any]:
    """Build one Lance-ready DiDeMo video row."""

    metadata = probe_video_bytes(video_bytes, source_path=video)
    return {
        "video_id": Path(video).stem,
        "video": video,
        "video_bytes": video_bytes,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _build_frame_unit_row(video: str, video_bytes: bytes, *, max_frames: int) -> dict[str, Any]:
    """Build one DiDeMo full-video frame-unit row."""

    return build_full_video_frame_unit_from_bytes(
        video_bytes=video_bytes,
        frame_unit_key=video,
        source_key=Path(video).stem,
        source_path=video,
        max_frames=max_frames,
    )


def _iter_video_rows_from_archives(
    *,
    archive_paths: Sequence[Path],
    needed_videos: set[str],
) -> Iterator[dict[str, Any]]:
    """Yield selected video rows from one tar or split-tar archive."""

    needed_members = {f"video/{video}" for video in needed_videos}
    with _open_tar_stream(archive_paths) as archive:
        for member in archive:
            if not member.isfile() or member.name not in needed_members:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read DiDeMo tar member: {member.name}")
            video = member.name.removeprefix("video/")
            yield _build_video_row(video, handle.read())


def _iter_frame_unit_rows_from_archives(
    *,
    archive_paths: Sequence[Path],
    needed_videos: set[str],
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
) -> Iterator[dict[str, Any]]:
    """Yield frame-unit rows from one tar or split-tar archive."""

    needed_members = {f"video/{video}" for video in needed_videos}
    found: set[str] = set()
    with _open_tar_stream(archive_paths) as archive:
        for member in archive:
            if not member.isfile() or member.name not in needed_members:
                continue
            video = member.name.removeprefix("video/")
            found.add(video)
            handle = archive.extractfile(member)
            if handle is None:
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="DiDeMo",
                        split="all",
                        source_key=Path(video).stem,
                        frame_unit_key=video,
                        reason="cannot_read_tar_member",
                        error=f"Cannot read DiDeMo tar member: {member.name}",
                        source_path=video,
                    )
                )
                continue
            try:
                row = _build_frame_unit_row(video, handle.read(), max_frames=full_video_max_frames)
            except Exception as error:
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="DiDeMo",
                        split="all",
                        source_key=Path(video).stem,
                        frame_unit_key=video,
                        reason="invalid_video",
                        error=str(error),
                        source_path=video,
                    )
                )
                continue
            valid_keys.add(video)
            yield row

    for missing_video in sorted(needed_videos - found):
        bad_media_rows.append(
            build_bad_media_row(
                dataset="DiDeMo",
                split="all",
                source_key=Path(missing_video).stem,
                frame_unit_key=missing_video,
                reason="missing_video",
                error="Missing DiDeMo video in tar archives",
                source_path=missing_video,
            )
        )


def _video_batches(
    *,
    train_archives: Sequence[Path],
    test_archive: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Build video batches by streaming tar members without unpacking files."""

    train_needed = {video for video in video_refs if video.startswith("train/")}
    test_needed = {video for video in video_refs if video.startswith("test/")}
    found: set[str] = set()
    batch: list[dict[str, Any]] = []
    for row in _iter_video_rows_from_archives(archive_paths=train_archives, needed_videos=train_needed):
        found.add(str(row["video"]))
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)
            batch = []
    for row in _iter_video_rows_from_archives(archive_paths=[test_archive], needed_videos=test_needed):
        found.add(str(row["video"]))
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)

    missing = sorted(set(video_refs) - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing DiDeMo videos in tar archives: {preview}")


def _frame_unit_batches(
    *,
    train_archives: Sequence[Path],
    test_archive: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Build pre-extracted frame-unit batches from split tar members."""

    train_needed = {video for video in video_refs if video.startswith("train/")}
    test_needed = {video for video in video_refs if video.startswith("test/")}
    batch: list[dict[str, Any]] = []
    for row in _iter_frame_unit_rows_from_archives(
        archive_paths=train_archives,
        needed_videos=train_needed,
        valid_keys=valid_keys,
        bad_media_rows=bad_media_rows,
        full_video_max_frames=full_video_max_frames,
    ):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=FRAME_UNIT_SCHEMA)
            batch = []
    for row in _iter_frame_unit_rows_from_archives(
        archive_paths=[test_archive],
        needed_videos=test_needed,
        valid_keys=valid_keys,
        bad_media_rows=bad_media_rows,
        full_video_max_frames=full_video_max_frames,
    ):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=FRAME_UNIT_SCHEMA)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=FRAME_UNIT_SCHEMA)


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)


def _bad_media_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield bad-media rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=BAD_MEDIA_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=BAD_MEDIA_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_didemo_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the DiDeMo runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing DiDeMo frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video")


def write_didemo_root(
    *,
    source_root: Path,
    train_archives: Sequence[Path],
    test_archive: Path,
    output_root: Path,
    overwrite: bool = False,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    full_video_max_frames: int = 64,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert DiDeMo into a raw-preserving Lance training root."""

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
                train_archives=train_archives,
                test_archive=test_archive,
                video_refs=video_refs,
                batch_size=video_batch_size,
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
            train_archives=train_archives,
            test_archive=test_archive,
            video_refs=valid_video_refs,
            batch_size=video_batch_size,
        ),
    )
    lance.write_dataset(
        video_reader,
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_didemo_split_rows(source_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video"] in valid_video_keys]
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
        _split_batches(split_rows["test"], batch_size=split_batch_size),
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

    ensure_didemo_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "splits": split_counts,
        "exclusion_rows": split_counts["test"],
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "didemo_train_frame_unit_lance_v1",
    }


__all__ = [
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_didemo_indices",
    "load_didemo_split_rows",
    "write_didemo_root",
]
