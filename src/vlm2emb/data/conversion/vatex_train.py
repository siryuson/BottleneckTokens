"""VATEX training conversion utilities."""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import tarfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
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
CAPTION_LIST_TYPE = pa.list_(pa.string())
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("en_caption", CAPTION_LIST_TYPE),
        pa.field("ch_caption", CAPTION_LIST_TYPE),
    ]
)
VIDEOS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video", pa.string()),
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


def normalize_vatex_video_member(value: object) -> str:
    """Normalize a VATEX video path to the tar member key used by qingy2024/VaTeX."""

    raw = str(value).replace("\\", "/").strip()
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        raise ValueError("VATEX video member path cannot be empty.")
    if raw.endswith(".mp4") and raw.startswith("vatex-dataset/"):
        return raw
    stem = Path(raw).stem if raw.endswith(".mp4") else raw
    return f"vatex-dataset/{stem}.mp4"


def _load_annotation_rows(annotation_path: Path) -> list[dict[str, Any]]:
    """Load VATEX training annotation rows from the official JSON file."""

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing VATEX annotation file: {annotation_path}")
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"VATEX annotation must contain a JSON list: {annotation_path}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"VATEX row {index} must be an object.")
        video_id = str(row["videoID"])
        en_caption = row.get("enCap")
        ch_caption = row.get("chCap")
        if not isinstance(en_caption, list) or not en_caption:
            raise TypeError(f"VATEX row enCap must be a non-empty list: videoID={video_id!r}")
        if not isinstance(ch_caption, list):
            raise TypeError(f"VATEX row chCap must be a list: videoID={video_id!r}")
        normalized.append(
            {
                "video_id": video_id,
                "video": normalize_vatex_video_member(video_id),
                "en_caption": [str(caption) for caption in en_caption],
                "ch_caption": [str(caption) for caption in ch_caption],
            }
        )
    return normalized


def load_vatex_split_rows(annotation_path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Return VATEX train and eval-excluded train split rows."""

    rows = _load_annotation_rows(annotation_path)
    return OrderedDict(
        [
            ("official_train", rows),
            ("official_train_without_mmeb_v2_eval", list(rows)),
        ]
    )


def collect_video_refs(annotation_path: Path) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading video bytes."""

    rows = _load_annotation_rows(annotation_path)
    video_by_id: OrderedDict[str, str] = OrderedDict()
    for row in rows:
        video_by_id.setdefault(str(row["video_id"]), str(row["video"]))
    counts = {split_name: len(rows) for split_name in SUPPORTED_SPLITS}
    return video_by_id, counts


def _build_video_row(video_id: str, video: str, video_bytes: bytes) -> dict[str, Any]:
    """Build one Lance-ready VATEX video row."""

    metadata = probe_video_bytes(video_bytes, source_path=video)
    return {
        "video_id": video_id,
        "video": video,
        "video_bytes": video_bytes,
        "fps": float(metadata["fps"]),
        "total_num_frames": int(metadata["total_num_frames"]),
    }


def _iter_video_rows_from_tar(
    *,
    video_archive: Path,
    video_refs: OrderedDict[str, str],
    bad_video_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield selected video rows from the VATEX tar archive."""

    found: set[str] = set()
    bad_video_ids = bad_video_ids if bad_video_ids is not None else set()
    needed_by_member = {normalize_vatex_video_member(video): video_id for video_id, video in video_refs.items()}
    with tarfile.open(video_archive, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized_member_name = normalize_vatex_video_member(member.name)
            video_id = needed_by_member.get(normalized_member_name)
            if video_id is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read VATEX tar member: {member.name}")
            found.add(normalized_member_name)
            try:
                yield _build_video_row(video_id, normalized_member_name, handle.read())
            except Exception:
                bad_video_ids.add(video_id)

    missing = sorted(set(needed_by_member) - found)
    for member_name in missing:
        bad_video_ids.add(needed_by_member[member_name])


def _iter_frame_unit_rows_from_tar(
    *,
    video_archive: Path,
    video_refs: OrderedDict[str, str],
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
    frame_unit_workers: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield selected full-video frame-units from the VATEX tar archive."""

    found: set[str] = set()
    needed_by_member = {normalize_vatex_video_member(video): video_id for video_id, video in video_refs.items()}
    workers = max(1, int(frame_unit_workers))

    if workers <= 1:
        yield from _iter_frame_unit_rows_from_tar_inline(
            video_archive=video_archive,
            video_refs=video_refs,
            valid_keys=valid_keys,
            bad_media_rows=bad_media_rows,
            full_video_max_frames=full_video_max_frames,
        )
        return

    with tarfile.open(video_archive, mode="r:*") as archive:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            pending: set[Future[tuple[str, dict[str, Any]]]] = set()
            max_pending = workers
            for member in archive:
                if not member.isfile():
                    continue
                normalized_member_name = normalize_vatex_video_member(member.name)
                video_id = needed_by_member.get(normalized_member_name)
                if video_id is None:
                    continue
                found.add(normalized_member_name)
                handle = archive.extractfile(member)
                if handle is None:
                    bad_media_rows.append(
                        build_bad_media_row(
                            dataset="VATEX",
                            split="all",
                            source_key=video_id,
                            frame_unit_key=normalized_member_name,
                            reason="invalid_video",
                            error=f"Cannot read VATEX tar member: {member.name}",
                            source_path=normalized_member_name,
                        )
                    )
                    continue
                pending.add(
                    executor.submit(
                        _build_frame_unit_worker,
                        video_id,
                        normalized_member_name,
                        handle.read(),
                        full_video_max_frames,
                    )
                )
                while len(pending) >= max_pending:
                    yield from _drain_frame_unit_futures(
                        pending,
                        valid_keys=valid_keys,
                        bad_media_rows=bad_media_rows,
                        wait_for_completion=True,
                    )
            while pending:
                yield from _drain_frame_unit_futures(
                    pending,
                    valid_keys=valid_keys,
                    bad_media_rows=bad_media_rows,
                    wait_for_completion=True,
                )

    missing = sorted(set(needed_by_member) - found)
    for member_name in missing:
        bad_media_rows.append(
            build_bad_media_row(
                dataset="VATEX",
                split="all",
                source_key=needed_by_member[member_name],
                frame_unit_key=member_name,
                reason="missing_video",
                error=f"Missing VATEX tar member: {member_name}",
                source_path=member_name,
            )
        )


def _iter_frame_unit_rows_from_tar_inline(
    *,
    video_archive: Path,
    video_refs: OrderedDict[str, str],
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
) -> Iterator[dict[str, Any]]:
    """Yield frame-unit rows from the VATEX tar archive in the current process."""

    found: set[str] = set()
    needed_by_member = {normalize_vatex_video_member(video): video_id for video_id, video in video_refs.items()}
    with tarfile.open(video_archive, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized_member_name = normalize_vatex_video_member(member.name)
            video_id = needed_by_member.get(normalized_member_name)
            if video_id is None:
                continue
            found.add(normalized_member_name)
            handle = archive.extractfile(member)
            if handle is None:
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="VATEX",
                        split="all",
                        source_key=video_id,
                        frame_unit_key=normalized_member_name,
                        reason="invalid_video",
                        error=f"Cannot read VATEX tar member: {member.name}",
                        source_path=normalized_member_name,
                    )
                )
                continue
            status, payload = _build_frame_unit_worker(
                video_id,
                normalized_member_name,
                handle.read(),
                full_video_max_frames,
            )
            if status == "ok":
                valid_keys.add(normalized_member_name)
                yield payload
            else:
                bad_media_rows.append(payload)

    missing = sorted(set(needed_by_member) - found)
    for member_name in missing:
        bad_media_rows.append(
            build_bad_media_row(
                dataset="VATEX",
                split="all",
                source_key=needed_by_member[member_name],
                frame_unit_key=member_name,
                reason="missing_video",
                error=f"Missing VATEX tar member: {member_name}",
                source_path=member_name,
            )
        )


def _build_frame_unit_worker(
    video_id: str,
    normalized_member_name: str,
    video_bytes: bytes,
    full_video_max_frames: int,
) -> tuple[str, dict[str, Any]]:
    """Build one VATEX frame-unit row in a worker process."""

    try:
        row = build_full_video_frame_unit_from_bytes(
            video_bytes=video_bytes,
            frame_unit_key=normalized_member_name,
            source_key=video_id,
            source_path=normalized_member_name,
            max_frames=full_video_max_frames,
        )
    except Exception as error:
        return (
            "bad",
            build_bad_media_row(
                dataset="VATEX",
                split="all",
                source_key=video_id,
                frame_unit_key=normalized_member_name,
                reason="invalid_video",
                error=str(error),
                source_path=normalized_member_name,
            ),
        )
    return "ok", row


def _drain_frame_unit_futures(
    pending: set[Future[tuple[str, dict[str, Any]]]],
    *,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    wait_for_completion: bool,
) -> Iterator[dict[str, Any]]:
    """Drain completed frame-unit futures and update parent-side audit sets."""

    done, remaining = wait(pending, return_when=FIRST_COMPLETED if wait_for_completion else FIRST_COMPLETED)
    pending.clear()
    pending.update(remaining)
    for future in done:
        status, payload = future.result()
        frame_unit_key = str(payload["frame_unit_key"])
        if status == "ok":
            valid_keys.add(frame_unit_key)
            yield payload
        else:
            bad_media_rows.append(payload)


def _video_batches(
    *,
    video_archive: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    bad_video_ids: set[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Build video table batches by streaming the tar archive once."""

    batch: list[dict[str, Any]] = []
    for row in _iter_video_rows_from_tar(
        video_archive=video_archive,
        video_refs=video_refs,
        bad_video_ids=bad_video_ids,
    ):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=VIDEOS_SCHEMA)


def _frame_unit_batches(
    *,
    video_archive: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    valid_keys: set[str],
    bad_media_rows: list[dict[str, Any]],
    full_video_max_frames: int,
    frame_unit_workers: int = 1,
) -> Iterator[pa.RecordBatch]:
    """Build frame-unit batches by streaming the tar archive once."""

    batch: list[dict[str, Any]] = []
    for row in _iter_frame_unit_rows_from_tar(
        video_archive=video_archive,
        video_refs=video_refs,
        valid_keys=valid_keys,
        bad_media_rows=bad_media_rows,
        full_video_max_frames=full_video_max_frames,
        frame_unit_workers=frame_unit_workers,
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


def ensure_vatex_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the VATEX runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if not frame_units_path.exists():
        raise FileNotFoundError(f"Missing VATEX frame-unit table: {frame_units_path}")
    _create_scalar_index(frame_units_path, "frame_unit_key")
    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_id")


def write_vatex_root(
    *,
    annotation_path: Path,
    video_archive: Path,
    output_root: Path,
    overwrite: bool = False,
    video_batch_size: int = 16,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    frame_unit_workers: int = 1,
    write_videos_table: bool = True,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert VATEX into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    video_refs, split_counts = collect_video_refs(annotation_path)
    valid_video_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                video_archive=video_archive,
                video_refs=video_refs,
                batch_size=video_batch_size,
                valid_keys=valid_video_keys,
                bad_media_rows=bad_media_rows,
                full_video_max_frames=full_video_max_frames,
                frame_unit_workers=frame_unit_workers,
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
        if normalize_vatex_video_member(video) in valid_video_keys
    )
    if write_videos_table:
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                VIDEOS_SCHEMA,
                _video_batches(
                    video_archive=video_archive,
                    video_refs=valid_video_refs,
                    batch_size=video_batch_size,
                ),
            ),
            str(output_root / "data" / "videos.lance"),
            schema=VIDEOS_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    split_rows = load_vatex_split_rows(annotation_path)
    split_rows = OrderedDict(
        (
            split_name,
            [
                row
                for row in rows
                if normalize_vatex_video_member(row["video"]) in valid_video_keys
            ],
        )
        for split_name, rows in split_rows.items()
    )
    split_counts = {split_name: len(rows) for split_name, rows in split_rows.items()}
    for split_name, rows in split_rows.items():
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(rows, batch_size=split_batch_size)),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches([], batch_size=split_batch_size)),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(BAD_MEDIA_SCHEMA, _bad_media_batches(bad_media_rows, batch_size=split_batch_size)),
        str(output_root / "data" / "exclusions" / "bad_media.lance"),
        schema=BAD_MEDIA_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_vatex_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "splits": split_counts,
        "exclusion_rows": 0,
        "bad_media_rows": len(bad_media_rows),
        "videos_table_written": write_videos_table,
        "frame_unit_workers": frame_unit_workers,
        "artifact_format": "vatex_train_frame_unit_lance_v1",
    }


__all__ = [
    "CAPTION_LIST_TYPE",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "collect_video_refs",
    "ensure_vatex_indices",
    "load_vatex_split_rows",
    "normalize_vatex_video_member",
    "write_vatex_root",
]
