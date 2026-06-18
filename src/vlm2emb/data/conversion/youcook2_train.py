"""YouCook2 training conversion utilities."""

from __future__ import annotations

import io
import shutil
import tarfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE
from vlm2emb.data.conversion.video_frame_units import (
    BAD_MEDIA_SCHEMA,
    FRAME_UNIT_SCHEMA,
    build_bad_media_row,
    build_row_frame_unit_from_rgb_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes
from vlm2emb.data.utils.video_paths import normalize_youcook2_video_path

SUPPORTED_SPLITS = ("official_train", "official_train_without_mmeb_v2_eval")
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_path", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("recipe_type", pa.int32()),
        pa.field("id", pa.int32()),
        pa.field("num_frames", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("width", pa.int32()),
        pa.field("channels", pa.int32()),
        pa.field("frame_unit_key", pa.string()),
    ]
)
VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video_path", pa.string()),
        pa.field("video_id", pa.string()),
        pa.field("video_bytes", pa.binary()),
        pa.field("fps", pa.float32()),
        pa.field("total_num_frames", pa.int32()),
    ]
)
PARQUET_READ_COLUMNS = (
    "video_path",
    "caption",
    "recipe_type",
    "id",
    "num_frames",
    "height",
    "width",
    "channels",
)
PARQUET_FRAME_READ_COLUMNS = (*PARQUET_READ_COLUMNS, "binary_frames")


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


def _parquet_files(parquet_root: Path) -> list[Path]:
    """Return sorted YouCook2 train parquet shards."""

    data_root = parquet_root / "data" if (parquet_root / "data").is_dir() else parquet_root
    files = sorted(data_root.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing YouCook2 parquet shards: {data_root}/train-*.parquet")
    return files


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one row while keeping the original train-view columns."""

    normalized_video_path = normalize_youcook2_video_path(str(row["video_path"]))
    row_id = int(row.get("id", 0) or 0)
    return {
        "video_path": str(row["video_path"]),
        "caption": str(row.get("caption", "") or ""),
        "recipe_type": int(row.get("recipe_type", 0) or 0),
        "id": row_id,
        "num_frames": int(row.get("num_frames", 0) or 0),
        "height": int(row.get("height", 0) or 0),
        "width": int(row.get("width", 0) or 0),
        "channels": int(row.get("channels", 0) or 0),
        "frame_unit_key": _frame_unit_key(normalized_video_path, row_id=row_id),
    }


def _frame_unit_key(video_path: str, *, row_id: int) -> str:
    """Build a stable row-level frame-unit key."""

    return f"{video_path}#row={row_id}"


def iter_youcook2_train_rows(parquet_root: Path) -> Iterator[dict[str, Any]]:
    """Yield YouCook2 train rows from all parquet shards without binary frames."""

    for path in _parquet_files(parquet_root):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=list(PARQUET_READ_COLUMNS)):
            for row in pa.Table.from_batches([batch]).to_pylist():
                yield _normalize_row(row)


def _iter_youcook2_frame_rows(parquet_root: Path) -> Iterator[dict[str, Any]]:
    """Yield normalized YouCook2 rows with row-level frame tensors."""

    for path in _parquet_files(parquet_root):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=list(PARQUET_FRAME_READ_COLUMNS)):
            for row in pa.Table.from_batches([batch]).to_pylist():
                normalized = _normalize_row(row)
                normalized["binary_frames"] = row.get("binary_frames")
                yield normalized


def load_youcook2_split_rows(parquet_root: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Return official train and eval-excluded train rows."""

    rows = list(iter_youcook2_train_rows(parquet_root))
    return OrderedDict(
        [
            ("official_train", rows),
            ("official_train_without_mmeb_v2_eval", list(rows)),
        ]
    )


def collect_video_refs(parquet_root: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique normalized video refs and split row counts."""

    video_ids_by_path: OrderedDict[str, str] = OrderedDict()
    rows = list(iter_youcook2_train_rows(parquet_root))
    for row in rows:
        video_path = normalize_youcook2_video_path(str(row["video_path"]))
        video_ids_by_path.setdefault(video_path, Path(video_path).stem)
    counts = {split_name: len(rows) for split_name in SUPPORTED_SPLITS}
    return video_ids_by_path, counts


def _open_tar_stream(paths: Sequence[Path]) -> tarfile.TarFile:
    """Open one tar or split-tar sequence for streaming reads."""

    if not paths:
        raise ValueError("At least one YouCook2 video archive path is required.")
    normalized = [Path(path) for path in paths]
    if len(normalized) == 1:
        return tarfile.open(normalized[0], mode="r:*")
    return tarfile.open(fileobj=ConcatenatedFile(normalized), mode="r|*")


def _build_video_row(video_path: str, video_bytes: bytes) -> dict[str, Any]:
    """Build one Lance-ready video row."""

    metadata = probe_video_bytes(video_bytes, source_path=video_path)
    return {
        "video_path": video_path,
        "video_id": Path(video_path).stem,
        "video_bytes": video_bytes,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _iter_video_rows_from_archives(
    *,
    archive_paths: Sequence[Path],
    needed_videos: set[str],
) -> Iterator[dict[str, Any]]:
    """Yield selected video rows from split tar archives."""

    found: set[str] = set()
    with _open_tar_stream(archive_paths) as archive:
        for member in archive:
            normalized_member_name = normalize_youcook2_video_path(member.name)
            if not member.isfile() or normalized_member_name not in needed_videos:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read YouCook2 tar member: {member.name}")
            found.add(normalized_member_name)
            yield _build_video_row(normalized_member_name, handle.read())

    missing = sorted(needed_videos - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing YouCook2 videos in tar archives: {preview}")


def _video_batches(
    *,
    archive_paths: Sequence[Path],
    video_refs: OrderedDict[str, str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Build video table batches by streaming split tar archives once."""

    batch: list[dict[str, Any]] = []
    for row in _iter_video_rows_from_archives(archive_paths=archive_paths, needed_videos=set(video_refs)):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SPLIT_SCHEMA)


def _frame_unit_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield frame-unit rows as Arrow batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=FRAME_UNIT_SCHEMA)


def _bad_media_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield bad-media rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=BAD_MEDIA_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=BAD_MEDIA_SCHEMA)


def _build_frame_units(
    parquet_root: Path,
    *,
    segment_max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Build YouCook2 row-level frame-units and bad-media rows."""

    frame_units: list[dict[str, Any]] = []
    bad_media: list[dict[str, Any]] = []
    valid_keys: set[str] = set()
    for row in _iter_youcook2_frame_rows(parquet_root):
        frame_unit_key = str(row["frame_unit_key"])
        source_path = normalize_youcook2_video_path(str(row["video_path"]))
        binary_frames = row.get("binary_frames")
        try:
            if not isinstance(binary_frames, (bytes, bytearray)):
                raise TypeError("missing binary_frames bytes")
            frame_unit = build_row_frame_unit_from_rgb_bytes(
                frame_unit_key=frame_unit_key,
                source_key=source_path,
                source_path=source_path,
                binary_frames=bytes(binary_frames),
                num_frames=int(row["num_frames"]),
                height=int(row["height"]),
                width=int(row["width"]),
                channels=int(row["channels"]),
                max_frames=segment_max_frames,
            )
        except Exception as error:
            bad_media.append(
                build_bad_media_row(
                    dataset="YouCook2",
                    split="official_train",
                    source_key=source_path,
                    frame_unit_key=frame_unit_key,
                    reason="invalid_row_frames",
                    error=str(error),
                    source_path=source_path,
                )
            )
            continue
        frame_units.append(frame_unit)
        valid_keys.add(frame_unit_key)
    return frame_units, bad_media, valid_keys


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_youcook2_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the YouCook2 runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing YouCook2 frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_path")


def write_youcook2_root(
    *,
    parquet_root: Path,
    video_archives: Sequence[Path],
    output_root: Path,
    overwrite: bool = False,
    video_batch_size: int = 16,
    split_batch_size: int = 4096,
    segment_max_frames: int = 8,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert YouCook2 into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    frame_units, bad_media_rows, valid_frame_unit_keys = _build_frame_units(
        parquet_root,
        segment_max_frames=segment_max_frames,
    )
    video_refs, split_counts = collect_video_refs(parquet_root)
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            VIDEOS_SCHEMA,
            _video_batches(
                archive_paths=video_archives,
                video_refs=video_refs,
                batch_size=video_batch_size,
            ),
        ),
        str(output_root / "data" / "videos.lance"),
        schema=VIDEOS_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(frame_units, batch_size=split_batch_size),
        ),
        str(output_root / "data" / "frames.lance"),
        schema=FRAME_UNIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    split_rows = load_youcook2_split_rows(parquet_root)
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

    ensure_youcook2_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(video_refs),
        "frame_unit_rows": len(frame_units),
        "splits": split_counts,
        "exclusion_rows": 0,
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "youcook2_train_frame_unit_lance_v1",
    }


__all__ = [
    "PARQUET_READ_COLUMNS",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_youcook2_indices",
    "iter_youcook2_train_rows",
    "load_youcook2_split_rows",
    "normalize_youcook2_video_path",
    "write_youcook2_root",
]
