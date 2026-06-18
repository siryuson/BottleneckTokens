"""HMDB51 training conversion utilities."""

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
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
)
from vlm2emb.data.utils.video import probe_video_bytes

SUPPORTED_SPLITS = (
    "official_split1_train",
    "official_split1_test",
    "official_split2_train",
    "official_split2_test",
    "official_split3_train",
    "official_split3_test",
    "vlm2vec_train",
    "vlm2vec_test_1k",
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
OFFICIAL_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("video_path", pa.string()),
        pa.field("label", pa.string()),
        pa.field("split_tag", pa.int8()),
        pa.field("split_file", pa.string()),
    ]
)
VLM2VEC_SPLIT_SCHEMA = pa.schema(
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


def _normalize_vlm2vec_row(row: dict[str, Any]) -> dict[str, str]:
    """Normalize one VLM2Vec metadata row while preserving source fields."""

    return {
        "video_id": str(row["video_id"]),
        "video_path": str(row["video_path"]),
        "pos_text": str(row["pos_text"]),
    }


def _normalize_official_row(*, class_name: str, filename: str, split_tag: int, split_file: str) -> dict[str, Any]:
    """Normalize one HMDB51 official split row."""

    video_path = f"{class_name}/{filename.strip()}"
    return {
        "video_id": Path(video_path).stem,
        "video_path": video_path,
        "label": class_name,
        "split_tag": split_tag,
        "split_file": split_file,
    }


def _load_official_split_rows(raw_root: Path, split_idx: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load one official HMDB51 split into train and test rows."""

    split_root = raw_root / "test_train_splits" / "testTrainMulti_7030_splits"
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing HMDB51 official split directory: {split_root}")

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for path in sorted(split_root.glob(f"*_test_split{split_idx}.txt")):
        class_name = path.name.replace(f"_test_split{split_idx}.txt", "")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid HMDB51 split row in {path} at line {line_number}: {line!r}")
            filename, tag_text = parts[:2]
            split_tag = int(tag_text)
            row = _normalize_official_row(
                class_name=class_name,
                filename=filename,
                split_tag=split_tag,
                split_file=path.name,
            )
            if split_tag == 1:
                train_rows.append(row)
            elif split_tag == 2:
                test_rows.append(row)
            elif split_tag == 0:
                continue
            else:
                raise ValueError(f"Unsupported HMDB51 split tag {split_tag!r} in {path} at line {line_number}.")
    return train_rows, test_rows


def load_hmdb51_split_rows(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
) -> OrderedDict[str, Iterable[dict[str, Any]]]:
    """Return split row iterables for official and VLM2Vec HMDB51 views."""

    split_rows: OrderedDict[str, Iterable[dict[str, Any]]] = OrderedDict()
    for split_idx in (1, 2, 3):
        train_rows, test_rows = _load_official_split_rows(raw_root, split_idx)
        split_rows[f"official_split{split_idx}_train"] = train_rows
        split_rows[f"official_split{split_idx}_test"] = test_rows
    split_rows["vlm2vec_train"] = (
        _normalize_vlm2vec_row(row) for row in _read_jsonl(vlm2vec_root / "data" / "train.jsonl")
    )
    split_rows["vlm2vec_test_1k"] = (
        _normalize_vlm2vec_row(row) for row in _read_jsonl(vlm2vec_root / "data" / "test.jsonl")
    )
    return split_rows


def collect_video_refs(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
) -> tuple[OrderedDict[str, str], dict[str, int]]:
    """Collect unique video refs and split row counts without reading video bytes."""

    video_id_by_path: OrderedDict[str, str] = OrderedDict()
    counts: dict[str, int] = {}
    for split_name, rows in load_hmdb51_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root).items():
        count = 0
        for row in rows:
            video_path = str(row["video_path"])
            video_id_by_path.setdefault(video_path, str(row["video_id"]))
            count += 1
        counts[split_name] = count
    return video_id_by_path, counts


def _resolve_raw_video_path(video_root: Path, relative_video_path: str) -> Path:
    """Resolve one HMDB51 video path under the extracted video root."""

    return video_root / relative_video_path


def _build_video_row(video_root: Path, item: tuple[str, str]) -> dict[str, Any]:
    """Read one raw video and return a Lance-ready row."""

    video_path, video_id = item
    raw_video_path = _resolve_raw_video_path(video_root, video_path)
    if not raw_video_path.is_file():
        raise FileNotFoundError(f"Missing raw HMDB51 video for {video_path}: {raw_video_path}")
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
    video_root: Path,
    video_refs: OrderedDict[str, str],
    batch_size: int,
    num_workers: int,
) -> Iterator[pa.RecordBatch]:
    """Build video table batches with bounded parallel file IO/probing."""

    workers = max(1, int(num_workers))
    items = list(video_refs.items())
    for chunk in _batched(items, batch_size):
        if workers == 1:
            rows = [_build_video_row(video_root, item) for item in chunk]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(lambda item: _build_video_row(video_root, item), chunk))
        yield pa.RecordBatch.from_pylist(rows, schema=VIDEOS_SCHEMA)


def _build_frame_unit_row(video_root: Path, item: tuple[str, str], *, max_frames: int) -> dict[str, Any]:
    """Read one raw video and build a frame-unit row."""

    video_path, video_id = item
    raw_video_path = _resolve_raw_video_path(video_root, video_path)
    if not raw_video_path.is_file():
        raise FileNotFoundError(f"Missing raw HMDB51 video for {video_path}: {raw_video_path}")
    return build_full_video_frame_unit_from_bytes(
        video_bytes=raw_video_path.read_bytes(),
        frame_unit_key=video_path,
        source_key=video_id,
        source_path=video_path,
        max_frames=max_frames,
    )


def _frame_unit_batches(
    *,
    video_root: Path,
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
                    results.append((item, _build_frame_unit_row(video_root, item, max_frames=full_video_max_frames)))
                except Exception as error:
                    results.append((item, error))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    (item, executor.submit(_build_frame_unit_row, video_root, item, max_frames=full_video_max_frames))
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
                        dataset="HMDB51",
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


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_hmdb51_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the HMDB51 runtime."""

    frame_units_path = output_root / "data" / "frames.lance"
    if frame_units_path.exists():
        _create_scalar_index(frame_units_path, "frame_unit_key")
        videos_path = output_root / "data" / "videos.lance"
        if videos_path.exists():
            _create_scalar_index(videos_path, "video_path")
        return

    videos_path = output_root / "data" / "videos.lance"
    if videos_path.exists():
        _create_scalar_index(videos_path, "video_path")
        return

    # Temporary compatibility for already-converted legacy roots. New
    # conversion writes only data/videos.lance.
    legacy_media_path = output_root / "data" / "media.lance"
    legacy_samples_path = output_root / "data" / "samples.lance"
    if not legacy_media_path.exists() or not legacy_samples_path.exists():
        raise FileNotFoundError(f"Missing HMDB51 video table under {output_root / 'data'}")
    _create_scalar_index(legacy_media_path, "asset_id")
    _create_scalar_index(legacy_samples_path, "sample_id")


def write_hmdb51_root(
    *,
    raw_root: Path,
    vlm2vec_root: Path,
    video_root: Path,
    output_root: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    video_batch_size: int = 64,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert HMDB51 into a raw-preserving Lance training root.

    The output contains one shared video table and one Lance table per source
    split. It intentionally does not write README, manifest, or dataset info
    files; those belong in human docs, not in the converted training artifact.
    """

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    video_refs, split_counts = collect_video_refs(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    valid_video_keys: set[str] = set()
    bad_media_rows: list[dict[str, Any]] = []
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _frame_unit_batches(
                video_root=video_root,
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

    split_rows = load_hmdb51_split_rows(raw_root=raw_root, vlm2vec_root=vlm2vec_root)
    for split_name, rows in split_rows.items():
        filtered_rows = [row for row in rows if row["video_path"] in valid_video_keys]
        split_counts[split_name] = len(filtered_rows)
        schema = VLM2VEC_SPLIT_SCHEMA if split_name.startswith("vlm2vec_") else OFFICIAL_SPLIT_SCHEMA
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

    ensure_hmdb51_indices(output_root)
    return {
        "output_root": str(output_root),
        "video_rows": len(valid_video_refs),
        "frame_unit_rows": len(valid_video_keys),
        "splits": split_counts,
        "bad_media_rows": len(bad_media_rows),
        "artifact_format": "hmdb51_train_frame_unit_lance_v1",
    }


__all__ = [
    "OFFICIAL_SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "VIDEOS_SCHEMA",
    "VLM2VEC_SPLIT_SCHEMA",
    "collect_video_refs",
    "ensure_hmdb51_indices",
    "load_hmdb51_split_rows",
    "write_hmdb51_root",
]
