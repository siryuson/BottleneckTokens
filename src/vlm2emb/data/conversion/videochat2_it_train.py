"""VideoChat2-IT training conversion utilities."""

from __future__ import annotations

import json
import re
import shutil
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.base import LanceDataset, LanceKeyedRowResolver
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE
from vlm2emb.data.conversion.video_frame_units import (
    BAD_MEDIA_SCHEMA,
    FRAME_UNIT_SCHEMA,
    FULL_VIDEO_MAX_FRAMES,
    build_bad_media_row,
    build_full_video_frame_unit_from_bytes,
)

SUPPORTED_SPLITS = ("official_train", "official_train_without_mmeb_v2_eval")
SUPPORTED_SUBSETS = ("next_qa", "youcook2", "ssv2", "k710", "videochatgpt")
SOURCE_FAMILY_BY_SUBSET = {
    "next_qa": "NExT-QA",
    "youcook2": "YouCook2",
    "ssv2": "Something-Something-V2",
    "k710": "Kinetics",
    "videochatgpt": "ActivityNet",
}
TASK_VIEW_BY_SUBSET = {
    "next_qa": "video_reasoning",
    "youcook2": "video_caption",
    "ssv2": "video_classification",
    "k710": "video_classification",
    "videochatgpt": "video_conversation",
}
ANNOTATION_REL_BY_SUBSET = {
    "next_qa": "video/reasoning/next_qa/train.json",
    "youcook2": "video/caption/youcook2/train.json",
    "ssv2": "video/classification/ssv2/train.json",
    "k710": "video/classification/k710/train.json",
    "videochatgpt": "video/conversation/videochatgpt/train.json",
}
FRAME_SOURCE_BY_SUBSET = {
    "next_qa": "nextqa",
    "youcook2": "youcook2",
    "ssv2": "ssv2",
    "k710": "kinetics700",
    "videochatgpt": "activitynet_full",
}
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("subset", pa.string()),
        pa.field("source_family", pa.string()),
        pa.field("task_view", pa.string()),
        pa.field("source_row_index", pa.int32()),
        pa.field("qa_index", pa.int32()),
        pa.field("video", pa.string()),
        pa.field("instruction", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("frame_unit_source", pa.string()),
        pa.field("frame_unit_key", pa.string()),
        pa.field("source_relative_path", pa.string()),
    ]
)


def _batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield fixed-size batches while preserving order."""

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
    """Load one VideoChat2-IT JSON annotation file."""

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"VideoChat2-IT annotation must be a JSON list: {path}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"VideoChat2-IT row {index} must be an object: {path}")
        normalized.append(row)
    return normalized


def _load_frame_keys(frame_path: Path) -> set[str]:
    """Load available frame-unit keys from one existing media root."""

    dataset = lance.dataset(str(frame_path))
    return set(dataset.to_table(columns=["frame_unit_key"]).column("frame_unit_key").to_pylist())


def _build_kinetics_key_map(frame_path: Path) -> dict[str, str]:
    """Map Kinetics clip basenames to their class-prefixed frame-unit keys."""

    dataset = lance.dataset(str(frame_path))
    values = dataset.to_table(columns=["frame_unit_key"]).column("frame_unit_key").to_pylist()
    return {PurePosixPath(value).name: value for value in values}


def _map_youcook2_frame_key(video: str) -> str | None:
    """Map a VideoChat2-IT YouCook2 split path to the local row-level key."""

    match = re.match(r"^(training/[^/]+/[^/]+)/split_(\d+)\.mp4$", video)
    if not match:
        return None
    return f"raw_videos/{match.group(1)}.mkv#row={int(match.group(2))}"


def _map_frame_key(
    *,
    subset: str,
    video: str,
    kinetics_by_basename: dict[str, str],
) -> str | None:
    """Return the local frame-unit key for one supported subset row."""

    if subset == "next_qa":
        return video.removesuffix(".mp4")
    if subset == "youcook2":
        return _map_youcook2_frame_key(video)
    if subset == "ssv2":
        return video
    if subset == "k710":
        return kinetics_by_basename.get(PurePosixPath(video).name)
    if subset == "videochatgpt":
        return PurePosixPath(video).stem
    raise ValueError(f"Unsupported VideoChat2-IT subset: {subset}")


def _qa_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized QA items from one source row."""

    items = row.get("QA", [])
    if not isinstance(items, list):
        raise TypeError(f"VideoChat2-IT row has non-list QA field: {row.get('video')!r}")
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"VideoChat2-IT QA item must be an object: {row.get('video')!r}")
        normalized.append(item)
    return normalized


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield VideoChat2-IT split rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=SPLIT_SCHEMA)


def _bad_media_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield bad-media rows as Arrow batches."""

    emitted = False
    for batch in _batched(rows, batch_size):
        emitted = True
        yield pa.RecordBatch.from_pylist(batch, schema=BAD_MEDIA_SCHEMA)
    if not emitted:
        yield pa.RecordBatch.from_pylist([], schema=BAD_MEDIA_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when missing."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_videochat2_it_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the VideoChat2-IT runtime."""

    data_root = output_root / "data"
    for split in SUPPORTED_SPLITS:
        split_path = data_root / f"{split}.lance"
        if split_path.exists():
            _create_scalar_index(split_path, "frame_unit_key")
    frame_path = data_root / "activitynet_full_frames.lance"
    if frame_path.exists():
        _create_scalar_index(frame_path, "frame_unit_key")


def _activitynet_video_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Return ordered ActivityNet video ids needed for VideoChatGPT rows."""

    ids: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        if row["frame_unit_source"] == "activitynet_full":
            ids.setdefault(str(row["frame_unit_key"]), None)
    return list(ids)


def _build_activitynet_frame_row(
    row: dict[str, Any],
    *,
    full_video_max_frames: int,
) -> dict[str, Any]:
    """Build one ActivityNet full-video frame-unit from an existing video row."""

    video_id = str(row["video_id"])
    return build_full_video_frame_unit_from_bytes(
        video_bytes=row["video_bytes"],
        frame_unit_key=video_id,
        source_key=video_id,
        source_path=str(row.get("member_path") or row.get("video") or video_id),
        max_frames=full_video_max_frames,
    )


def _activitynet_frame_batches(
    *,
    videos_lance: Path,
    video_ids: list[str],
    batch_size: int,
    num_workers: int,
    full_video_max_frames: int,
    bad_media_rows: list[dict[str, Any]],
) -> Iterator[pa.RecordBatch]:
    """Yield sampled ActivityNet full-video frame units for VideoChatGPT."""

    dataset = LanceDataset(str(videos_lance))
    resolver = LanceKeyedRowResolver(dataset, "video_id")
    workers = max(1, int(num_workers))
    for keys in _batched(video_ids, batch_size):
        source_rows = resolver.lookup_rows(
            keys,
            columns=["video", "video_id", "member_path", "video_bytes"],
            missing="skip",
        )
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for key, row in zip(keys, source_rows, strict=True):
                if row is None:
                    bad_media_rows.append(
                        build_bad_media_row(
                            dataset="VideoChat2-IT",
                            split="all",
                            source_key=key,
                            frame_unit_key=key,
                            reason="missing_activitynet_video",
                            error=f"Missing ActivityNet video_id: {key}",
                            source_path=key,
                        )
                    )
                    continue
                futures.append((key, row, executor.submit(
                    _build_activitynet_frame_row,
                    row,
                    full_video_max_frames=full_video_max_frames,
                )))
            for key, source_row, future in futures:
                try:
                    rows.append(future.result())
                except Exception as error:
                    bad_media_rows.append(
                        build_bad_media_row(
                            dataset="VideoChat2-IT",
                            split="all",
                            source_key=key,
                            frame_unit_key=key,
                            reason="invalid_activitynet_video",
                            error=str(error),
                            source_path=str(source_row.get("member_path") or key),
                        )
                    )
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=FRAME_UNIT_SCHEMA)
    if not video_ids:
        yield pa.RecordBatch.from_pylist([], schema=FRAME_UNIT_SCHEMA)


def load_videochat2_it_split_rows(
    *,
    annotation_root: Path,
    nextqa_frames: Path,
    youcook2_frames: Path,
    ssv2_frames: Path,
    kinetics700_frames: Path,
    activitynet_videos: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return media-backed split rows and bad-media audit rows."""

    available_keys = {
        "nextqa": _load_frame_keys(nextqa_frames),
        "youcook2": _load_frame_keys(youcook2_frames),
        "ssv2": _load_frame_keys(ssv2_frames),
    }
    kinetics_by_basename = _build_kinetics_key_map(kinetics700_frames)
    activitynet_dataset = lance.dataset(str(activitynet_videos))
    activitynet_ids = set(
        activitynet_dataset.to_table(columns=["video_id"]).column("video_id").to_pylist()
    )
    rows: list[dict[str, Any]] = []
    bad_media_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    for subset in SUPPORTED_SUBSETS:
        rel = ANNOTATION_REL_BY_SUBSET[subset]
        source_rows = _load_json_rows(annotation_root / rel)
        stats[subset] = {"source_rows": len(source_rows), "converted_rows": 0, "bad_media_rows": 0}
        frame_source = FRAME_SOURCE_BY_SUBSET[subset]
        for source_row_index, source_row in enumerate(source_rows):
            video = str(source_row.get("video", "") or "")
            frame_key = _map_frame_key(
                subset=subset,
                video=video,
                kinetics_by_basename=kinetics_by_basename,
            )
            if frame_source == "activitynet_full":
                has_media = frame_key in activitynet_ids
            elif frame_source == "kinetics700":
                has_media = frame_key is not None
            else:
                has_media = frame_key in available_keys[frame_source]
            if not frame_key or not has_media:
                stats[subset]["bad_media_rows"] += 1
                bad_media_rows.append(
                    build_bad_media_row(
                        dataset="VideoChat2-IT",
                        split="all",
                        source_key=video,
                        frame_unit_key=frame_key or "",
                        reason="missing_resolved_frame_unit",
                        error=f"Cannot resolve {subset} media for video={video!r}",
                        source_path=video,
                    )
                )
                continue
            for qa_index, qa in enumerate(_qa_items(source_row)):
                rows.append(
                    {
                        "id": f"{subset}:{source_row_index}:{qa_index}",
                        "subset": subset,
                        "source_family": SOURCE_FAMILY_BY_SUBSET[subset],
                        "task_view": TASK_VIEW_BY_SUBSET[subset],
                        "source_row_index": source_row_index,
                        "qa_index": qa_index,
                        "video": video,
                        "instruction": str(qa.get("i", "") or ""),
                        "question": str(qa.get("q", "") or ""),
                        "answer": str(qa.get("a", "") or ""),
                        "frame_unit_source": frame_source,
                        "frame_unit_key": frame_key,
                        "source_relative_path": rel,
                    }
                )
                stats[subset]["converted_rows"] += 1
    return rows, bad_media_rows, {"subsets": stats}


def write_videochat2_it_root(
    *,
    annotation_root: Path,
    output_root: Path,
    nextqa_frames: Path,
    youcook2_frames: Path,
    ssv2_frames: Path,
    kinetics700_frames: Path,
    activitynet_videos: Path,
    overwrite: bool = False,
    num_workers: int = 8,
    frame_batch_size: int = 16,
    split_batch_size: int = 4096,
    full_video_max_frames: int = FULL_VIDEO_MAX_FRAMES,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert media-backed VideoChat2-IT subsets into a formal Lance root."""

    data_root = output_root / "data"
    if data_root.exists():
        if not overwrite:
            raise FileExistsError(f"VideoChat2-IT data root already exists: {data_root}")
        shutil.rmtree(data_root)
    (data_root / "exclusions").mkdir(parents=True, exist_ok=True)

    rows, bad_media_rows, stats = load_videochat2_it_split_rows(
        annotation_root=annotation_root,
        nextqa_frames=nextqa_frames,
        youcook2_frames=youcook2_frames,
        ssv2_frames=ssv2_frames,
        kinetics700_frames=kinetics700_frames,
        activitynet_videos=activitynet_videos,
    )
    activitynet_ids = _activitynet_video_ids(rows)
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            FRAME_UNIT_SCHEMA,
            _activitynet_frame_batches(
                videos_lance=activitynet_videos,
                video_ids=activitynet_ids,
                batch_size=frame_batch_size,
                num_workers=num_workers,
                full_video_max_frames=full_video_max_frames,
                bad_media_rows=bad_media_rows,
            ),
        ),
        str(data_root / "activitynet_full_frames.lance"),
        schema=FRAME_UNIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    invalid_activitynet_keys = {
        row["frame_unit_key"]
        for row in bad_media_rows
        if row["reason"] in {"missing_activitynet_video", "invalid_activitynet_video"}
    }
    filtered_rows = [
        row
        for row in rows
        if row["frame_unit_source"] != "activitynet_full"
        or row["frame_unit_key"] not in invalid_activitynet_keys
    ]
    for split in SUPPORTED_SPLITS:
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(
                SPLIT_SCHEMA,
                _split_batches(filtered_rows, batch_size=split_batch_size),
            ),
            str(data_root / f"{split}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            BAD_MEDIA_SCHEMA,
            _bad_media_batches(bad_media_rows, batch_size=split_batch_size),
        ),
        str(data_root / "exclusions" / "bad_media.lance"),
        schema=BAD_MEDIA_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches([], batch_size=split_batch_size),
        ),
        str(data_root / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_videochat2_it_indices(output_root)
    summary = {
        "output_root": str(output_root),
        "splits": {split: len(filtered_rows) for split in SUPPORTED_SPLITS},
        "activitynet_full_frame_rows": len(activitynet_ids) - len(invalid_activitynet_keys),
        "bad_media_rows": len(bad_media_rows),
        "subsets": stats["subsets"],
        "artifact_format": "videochat2_it_media_backed_lance_v1",
    }
    (data_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ANNOTATION_REL_BY_SUBSET",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "SUPPORTED_SUBSETS",
    "ensure_videochat2_it_indices",
    "load_videochat2_it_split_rows",
    "write_videochat2_it_root",
]
