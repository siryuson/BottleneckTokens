"""Convert MMEB-V2 video evaluation datasets into Lance retrieval artifacts.

This module implements all video-side pipelines used by MMEB-V2 conversion,
including classification, question answering, text retrieval, moment retrieval,
and the MomentSeeker variant. It combines local MMEB-V2 frame assets with local
or HuggingFace annotation sources and writes normalized retrieval artifacts.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
from pyarrow.parquet import read_table
from tqdm.auto import tqdm

from .bindings import (
    get_source_by_role,
    load_hf_records,
    read_jsonl_records,
    resolve_source_path,
)
from .common import ensure_output_root, get_metadata_task_type, write_minimal_metadata
from .datasets import (
    VIDEO_MODALITY,
    get_modality,
    get_pipeline,
    get_spec,
    get_specs_by_pipeline,
)

FRAME_CACHE_SIZE = 128
MVBENCH_SUBSETS = (
    "episodic_reasoning",
    "action_sequence",
    "action_prediction",
    "action_antonym",
    "fine_grained_action",
    "unexpected_action",
    "object_existence",
    "object_interaction",
    "object_shuffle",
    "moving_direction",
    "action_localization",
    "scene_transition",
    "action_count",
    "moving_count",
    "moving_attribute",
    "state_change",
    "fine_grained_pose",
    "character_order",
    "egocentric_navigation",
    "counterfactual_inference",
)


def _write_lance(records: list[dict[str, Any]], path: Path) -> None:
    """Write one record list into a Lance dataset with inferred schema."""
    table = pa.Table.from_pylist(records)
    lance.write_dataset(table, str(path), mode="create")


def _write_metadata(output_path: Path, *, dataset_name: str) -> None:
    """Write the minimal metadata.json for one video artifact directory."""
    write_minimal_metadata(
        output_path,
        dataset_name=dataset_name,
        task_type=get_metadata_task_type(dataset_name, get_pipeline(dataset_name)),
        modality=get_modality(dataset_name),
    )


def _parallel_map_in_order(
    items: list[Any],
    fn,
    *,
    max_workers: int,
    desc: str,
):
    """Map work items in parallel while yielding results in input order."""
    total = len(items)
    if total == 0:
        return

    if max_workers <= 1:
        for item in tqdm(items, desc=desc):
            yield fn(item)
        return

    prefetch = min(total, max_workers * 2)
    item_iter = iter(items)
    pending = deque()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(prefetch):
            try:
                pending.append(executor.submit(fn, next(item_iter)))
            except StopIteration:
                break

        with tqdm(total=total, desc=desc) as progress:
            while pending:
                future = pending.popleft()
                yield future.result()
                progress.update(1)
                try:
                    pending.append(executor.submit(fn, next(item_iter)))
                except StopIteration:
                    continue


@lru_cache(maxsize=FRAME_CACHE_SIZE)
def _load_frame_dir_bytes_cached(frame_dir_str: str) -> tuple[bytes, ...]:
    """Read and cache every image frame under one extracted frame directory."""
    frame_dir = Path(frame_dir_str)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame path is not a directory: {frame_dir}")

    image_files = sorted(
        file_path
        for file_path in frame_dir.iterdir()
        if file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        raise FileNotFoundError(f"No frame images found in directory: {frame_dir}")

    frames: list[bytes] = []
    for image_file in image_files:
        frame_bytes = image_file.read_bytes()
        if not frame_bytes:
            raise ValueError(f"Frame file is empty: {image_file}")
        frames.append(frame_bytes)
    return tuple(frames)


def _load_video_frames(frame_root: Path, video_id: str) -> list[bytes]:
    """Load all extracted frames for one video id."""
    return list(_load_frame_dir_bytes_cached(str(frame_root / str(video_id))))


def _resolve_frame_root(
    dataset_name: str,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> Path:
    """Resolve the local MMEB-V2 frame root for one video dataset."""
    spec = get_spec(dataset_name)
    frames_source = get_source_by_role(spec, "frames")
    if frames_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing frames source in dataset list")
    return resolve_source_path(
        frames_source,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )


def _load_records_from_hf(
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Load video annotation rows from the declared HF dataset source."""
    spec = get_spec(dataset_name)
    annotation_source = get_source_by_role(spec, "annotation")
    if annotation_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing annotation source in dataset list")
    if annotation_source.kind != "hf_dataset":
        raise ValueError(f"[{dataset_name}] annotation source is not hf_dataset")

    if dataset_name == "MVBench":
        return load_hf_records(
            annotation_source.ref,
            subset_names=MVBENCH_SUBSETS,
            subset_field_name="_source_file",
        )

    return load_hf_records(annotation_source.ref)


def _load_annotation_records(
    dataset_name: str,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """Load video annotation rows from either HF or one local file source."""
    spec = get_spec(dataset_name)
    annotation_source = get_source_by_role(spec, "annotation")
    if annotation_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing annotation source in dataset list")

    if annotation_source.kind == "hf_dataset":
        return _load_records_from_hf(dataset_name)

    annotation_path = resolve_source_path(
        annotation_source,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    if annotation_path.suffix == ".jsonl":
        return read_jsonl_records(annotation_path)
    if annotation_path.suffix == ".parquet":
        return read_table(annotation_path).to_pylist()
    raise ValueError(f"[{dataset_name}] Unsupported annotation file: {annotation_path}")


def _build_first_seen_text_candidates(candidate_groups: list[list[str]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Build a first-seen global text candidate table from local groups."""
    candidate_id_by_text: dict[str, int] = {}
    candidate_records: list[dict[str, Any]] = []

    for candidates in candidate_groups:
        for candidate in candidates:
            if candidate not in candidate_id_by_text:
                candidate_id = len(candidate_records)
                candidate_id_by_text[candidate] = candidate_id
                candidate_records.append({"id": candidate_id, "text": candidate})

    return candidate_id_by_text, candidate_records


def _ensure_plain_list(value: Any) -> list[Any]:
    """Normalize one value into a plain Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate values without changing first-seen order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _validate_video_cls_candidates(dataset_name: str, candidates: list[str]) -> list[str]:
    """Reject blank classification labels instead of materializing them."""
    blank_indices = [
        candidate_index
        for candidate_index, candidate_text in enumerate(candidates)
        if not candidate_text.strip()
    ]
    if blank_indices:
        raise ValueError(
            f"[{dataset_name}] blank classification candidates are not allowed at indices {blank_indices}"
        )
    return candidates


def convert_video_cls_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> Path:
    """Convert one MMEB-V2 video classification dataset into rerun layout."""

    if get_pipeline(dataset_name) != "video_cls":
        raise ValueError(f"{dataset_name!r} is not a video_cls dataset")

    output_path = ensure_output_root(output_root / "video-tasks" / dataset_name)
    frame_root = _resolve_frame_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    records = _load_annotation_records(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )

    candidate_groups: list[list[str]] = []
    for row in records:
        pos_text = str(row["pos_text"])
        neg_text = [str(value) for value in row.get("neg_text", []) or []]
        if dataset_name == "SmthSmthV2":
            candidate_groups.append(
                _validate_video_cls_candidates(
                    dataset_name,
                    _dedupe_preserve_order([pos_text, *neg_text]),
                )
            )
        else:
            candidate_groups.append(
                _validate_video_cls_candidates(dataset_name, [pos_text])
            )

    candidate_id_by_text, candidate_records = _build_first_seen_text_candidates(candidate_groups)

    def _build_query(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        row_index, row = entry
        video_id = str(row["video_id"])
        query = dict(row)
        query["id"] = row_index
        query["video"] = _load_video_frames(frame_root, video_id)
        return query

    queries = list(
        _parallel_map_in_order(
            list(enumerate(records)),
            _build_query,
            max_workers=max_workers,
            desc=f"{dataset_name}: reading videos",
        )
    )

    qrels: list[dict[str, Any]] = []
    for query_id, candidates in enumerate(candidate_groups):
        if dataset_name == "SmthSmthV2":
            qrels.append(
                {
                    "query_id": query_id,
                    "mode": "exhaustive",
                    "candidate_ids": [candidate_id_by_text[candidate] for candidate in candidates],
                    "candidate_scores": [1.0] + [0.0] * (len(candidates) - 1),
                }
            )
        else:
            qrels.append(
                {
                    "query_id": query_id,
                    "mode": "sparse",
                    "candidate_ids": [candidate_id_by_text[candidates[0]]],
                    "candidate_scores": [1.0],
                }
            )

    _write_lance(queries, output_path / "queries.lance")
    _write_lance(candidate_records, output_path / "candidates.lance")
    _write_lance(qrels, output_path / "qrels.lance")
    _write_metadata(output_path, dataset_name=dataset_name)
    return output_path


def _resolve_video_qa_options_and_answer(
    dataset_name: str,
    row: dict[str, Any],
) -> tuple[list[str], int]:
    """Normalize one raw video-QA annotation row into options plus answer index.

    ``row`` is one original annotation record loaded from the dataset-specific
    source (HF or local file), not one converted Lance row. Different datasets
    store answer options in different raw shapes, so this helper extracts a
    common representation consumed by the shared conversion flow:

    - ``options``: ordered answer option texts
    - ``answer_idx``: zero-based index of the correct option
    """
    if dataset_name == "Video-MME":
        # Video-MME stores one explicit ``options`` list and the correct answer
        # as a letter in ``A``-``D``.
        options = [str(value) for value in _ensure_plain_list(row.get("options"))]
        answer_idx = ["A", "B", "C", "D"].index(str(row["answer"]))
        return _validate_video_qa_options(dataset_name, options), answer_idx
    if dataset_name == "NExTQA":
        # NExTQA stores options in positional fields ``a0``-``a4`` and the
        # answer as a zero-based integer index.
        options = ["" if row.get(f"a{option_idx}") is None else str(row.get(f"a{option_idx}")) for option_idx in range(5)]
        return _validate_video_qa_options(dataset_name, options), int(row["answer"])
    if dataset_name == "MVBench":
        # MVBench stores candidate texts in one list field and the answer as
        # the exact option text.
        options = ["" if value is None else str(value) for value in _ensure_plain_list(row.get("candidates"))]
        answer_text = "" if row.get("answer") is None else str(row.get("answer"))
        try:
            options = _validate_video_qa_options(dataset_name, options)
            return options, options.index(answer_text)
        except ValueError as exc:
            raise ValueError(f"MVBench answer not found in candidates: {answer_text!r}") from exc
    if dataset_name == "EgoSchema":
        # EgoSchema stores option texts in one list field and the answer as a
        # zero-based integer index.
        options = ["" if value is None else str(value) for value in _ensure_plain_list(row.get("option"))]
        return _validate_video_qa_options(dataset_name, options), int(row["answer"])
    if dataset_name == "ActivityNetQA":
        # ActivityNetQA is normalized into a binary yes/no option set.
        options = ["yes", "no"]
        answer_text = str(row["answer"]).strip().lower()
        try:
            return options, options.index(answer_text)
        except ValueError as exc:
            raise ValueError(f"ActivityNetQA answer must be yes/no, got {answer_text!r}") from exc
    raise NotImplementedError(f"Unsupported video_qa dataset: {dataset_name}")


def _validate_video_qa_options(dataset_name: str, options: list[str]) -> list[str]:
    """Reject blank answer options instead of silently materializing them."""
    blank_indices = [
        option_index
        for option_index, option_text in enumerate(options)
        if not option_text.strip()
    ]
    if blank_indices:
        raise ValueError(
            f"[{dataset_name}] blank answer options are not allowed at indices {blank_indices}"
        )
    return options


def _load_video_qa_query_frames(
    dataset_name: str,
    frame_root: Path,
    row: dict[str, Any],
) -> list[bytes]:
    """Load query video frames for one raw video-QA annotation row."""
    if dataset_name == "Video-MME":
        return _load_video_frames(frame_root, str(row["videoID"]))
    if dataset_name == "NExTQA":
        return _load_video_frames(frame_root, str(row["video"]))
    if dataset_name == "EgoSchema":
        return _load_video_frames(frame_root, str(row["video_idx"]))
    if dataset_name == "ActivityNetQA":
        return _load_video_frames(frame_root, f"v_{row['video_name']}")
    if dataset_name == "MVBench":
        subset_name = str(row["_source_file"])
        return _load_video_frames(frame_root / subset_name, str(row["video"]))
    raise NotImplementedError(f"Unsupported video_qa dataset: {dataset_name}")


def convert_video_qa_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> Path:
    """Convert one MMEB-V2 video QA dataset into rerun layout."""

    if get_pipeline(dataset_name) != "video_qa":
        raise ValueError(f"{dataset_name!r} is not a video_qa dataset")

    output_path = ensure_output_root(output_root / "video-tasks" / dataset_name)
    frame_root = _resolve_frame_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    records = _load_annotation_records(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )

    def _build_query(entry: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any], list[str], int]:
        row_index, row = entry
        options, answer_idx = _resolve_video_qa_options_and_answer(dataset_name, row)
        query = dict(row)
        query["id"] = row_index
        query["video"] = _load_video_qa_query_frames(dataset_name, frame_root, row)
        return query, options, answer_idx

    query_payloads = list(
        _parallel_map_in_order(
            list(enumerate(records)),
            _build_query,
            max_workers=max_workers,
            desc=f"{dataset_name}: reading videos",
        )
    )

    queries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    candidate_id_counter = 0

    for query, options, answer_idx in query_payloads:
        query_id = int(query["id"])
        queries.append(query)

        query_candidate_ids: list[int] = []
        for option_index, option_text in enumerate(options):
            candidate_id = candidate_id_counter
            candidate_id_counter += 1
            candidate_text = option_text
            if dataset_name == "MVBench":
                # MVBench keeps answer options as labeled multiple-choice text in
                # the candidate corpus so runtime can preserve the original
                # "(A)/(B)/..." surface instead of reconstructing labels later.
                candidate_text = f"({chr(65 + option_index)}) {option_text}"
            candidates.append(
                {
                    "id": candidate_id,
                    "text": candidate_text,
                }
            )
            query_candidate_ids.append(candidate_id)

        if answer_idx < 0 or answer_idx >= len(query_candidate_ids):
            raise ValueError(f"[{dataset_name}] invalid answer index {answer_idx} for query {query_id}")

        qrels.append(
            {
                "query_id": query_id,
                # MVBench is modeled as "hit the single correct answer text"
                # instead of "score the full local option pool with 0/1
                # exhaustively". The other video-QA subsets keep the full local
                # candidate set in qrels because the option pool itself is part
                # of the per-query evaluation contract.
                "mode": "sparse" if dataset_name == "MVBench" else "exhaustive",
                "candidate_ids": [query_candidate_ids[answer_idx]]
                if dataset_name == "MVBench"
                else list(query_candidate_ids),
                "candidate_scores": [1.0]
                if dataset_name == "MVBench"
                else [
                    1.0 if option_index == answer_idx else 0.0
                    for option_index in range(len(query_candidate_ids))
                ],
            }
        )

    _write_lance(queries, output_path / "queries.lance")
    _write_lance(candidates, output_path / "candidates.lance")
    _write_lance(qrels, output_path / "qrels.lance")
    _write_metadata(output_path, dataset_name=dataset_name)
    return output_path


def _video_ret_candidate_source_id(dataset_name: str, row: dict[str, Any]) -> str:
    """Return the stable source video identifier from one raw retrieval row.

    Each dataset stores the source video key in a different raw field. This
    helper extracts the identifier used for:

    - global candidate deduplication
    - frame directory lookup under MMEB-V2
    - sparse qrels construction
    """
    if dataset_name in {"MSR-VTT", "MSVD"}:
        return str(row["video_id"])
    if dataset_name == "DiDeMo":
        return Path(str(row["video"])).stem
    if dataset_name == "VATEX":
        return str(row["videoID"])
    if dataset_name == "YouCook2":
        return str(row["id"])
    raise NotImplementedError(f"Unsupported video_ret dataset: {dataset_name}")


def _build_video_ret_candidate_record(
    dataset_name: str,
    row: dict[str, Any],
    *,
    candidate_id: int,
    video_bytes: list[bytes],
) -> dict[str, Any]:
    """Build one converted candidate row from one raw retrieval annotation row.

    The candidate schema keeps only the fields required to identify the source
    video plus the loaded frame bytes. Dataset-specific raw identifiers are
    preserved because upstream retrieval subsets do not share one unified field
    name.
    """
    if dataset_name in {"MSR-VTT", "MSVD"}:
        return {
            "id": candidate_id,
            "video_id": row["video_id"],
            "video": video_bytes,
        }
    if dataset_name == "DiDeMo":
        record = {
            "id": candidate_id,
            "video": video_bytes,
            "video_path": row["video"],
        }
        if "source" in row:
            record["source"] = row["source"]
        return record
    if dataset_name == "VATEX":
        return {
            "id": candidate_id,
            "videoID": row["videoID"],
            "video": video_bytes,
        }
    if dataset_name == "YouCook2":
        record = {
            "id": candidate_id,
            "video": video_bytes,
            "source_id": row["id"],
        }
        for field_name in ("segment", "video_path", "youtube_id"):
            if field_name in row:
                record[field_name] = row[field_name]
        return record
    raise NotImplementedError(f"Unsupported video_ret dataset: {dataset_name}")


def convert_video_ret_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> Path:
    """Convert one MMEB-V2 video retrieval dataset into rerun layout."""

    if get_pipeline(dataset_name) != "video_text_retrieval":
        raise ValueError(f"{dataset_name!r} is not a video_text_retrieval dataset")

    output_path = ensure_output_root(output_root / "video-tasks" / dataset_name)
    frame_root = _resolve_frame_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    records = _load_annotation_records(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )

    queries: list[dict[str, Any]] = []
    candidate_info_by_source_id: dict[str, dict[str, Any]] = {}
    candidate_source_ids_in_order: list[str] = []

    for query_id, row in enumerate(records):
        query = dict(row)
        query["id"] = query_id
        queries.append(query)

        source_id = _video_ret_candidate_source_id(dataset_name, row)
        if source_id not in candidate_info_by_source_id:
            # One source video can appear under many captions/queries. Collapse
            # the corpus to the first occurrence of each source video so
            # candidate ids stay stable across reruns and qrels always point at
            # one global video pool rather than query-local duplicates.
            candidate_info_by_source_id[source_id] = dict(row)
            candidate_source_ids_in_order.append(source_id)

    candidate_id_by_source_id = {
        source_id: candidate_id
        for candidate_id, source_id in enumerate(candidate_source_ids_in_order)
    }
    qrels: list[dict[str, Any]] = []
    for query_id, row in enumerate(records):
        source_id = _video_ret_candidate_source_id(dataset_name, row)
        qrels.append(
            {
                "query_id": query_id,
                "mode": "sparse",
                "candidate_ids": [candidate_id_by_source_id[source_id]],
                "candidate_scores": [1.0],
            }
        )

    def _build_candidate(entry: tuple[int, str]) -> dict[str, Any]:
        candidate_id, source_id = entry
        row = candidate_info_by_source_id[source_id]
        video_bytes = _load_video_frames(frame_root, source_id)
        return _build_video_ret_candidate_record(
            dataset_name,
            row,
            candidate_id=candidate_id,
            video_bytes=video_bytes,
        )

    candidates = list(
        _parallel_map_in_order(
            list(enumerate(candidate_source_ids_in_order)),
            _build_candidate,
            max_workers=max_workers,
            desc=f"{dataset_name}: reading videos",
        )
    )

    _write_lance(queries, output_path / "queries.lance")
    _write_lance(candidates, output_path / "candidates.lance")
    _write_lance(qrels, output_path / "qrels.lance")
    _write_metadata(output_path, dataset_name=dataset_name)
    return output_path


def _load_video_mret_query_frames(frame_root: Path, row: dict[str, Any]) -> list[bytes]:
    """Load query frames for one local moment-retrieval row.

    Current MMEB-V2 moment-retrieval subsets identify one per-query clip
    directory through ``clips_dir_path``. The final path component is reused as
    the extracted local clip name under ``frame_root/<clip_name>/query``.
    """
    clips_dir_path = str(row.get("clips_dir_path", "") or "")
    if not clips_dir_path:
        raise ValueError("Missing clips_dir_path for video_mret query")
    clip_name = Path(clips_dir_path).name
    query_dir = frame_root / clip_name / "query"
    return list(_load_frame_dir_bytes_cached(str(query_dir)))


def _load_video_mret_directory_candidates(clip_dir: Path) -> tuple[list[list[bytes]], list[int]]:
    """Load candidates from the explicit clip-directory layout.

    Example:
    - ``positive_clip/``
    - ``negative_clip_0/``
    - ``negative_clip_1/``
    """
    candidate_videos: list[list[bytes]] = [
        list(_load_frame_dir_bytes_cached(str(clip_dir / "positive_clip")))
    ]
    positive_indices = [0]
    neg_idx = 0
    while (clip_dir / f"negative_clip_{neg_idx}").exists():
        candidate_videos.append(
            list(_load_frame_dir_bytes_cached(str(clip_dir / f"negative_clip_{neg_idx}")))
        )
        neg_idx += 1
    return candidate_videos, positive_indices


def _load_video_mret_flat_candidates(clip_dir: Path) -> tuple[list[list[bytes]], list[int]]:
    """Load candidates from the flat extracted-frame layout.

    Example file names:
    - ``positive_000.jpg``
    - ``negative_0_000.jpg``
    - ``negative_1_000.jpg``
    """
    positive_frames: list[bytes] = []
    negative_frames: dict[int, list[bytes]] = {}
    for frame_file in sorted(clip_dir.iterdir()):
        if frame_file.is_dir():
            continue
        if frame_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame_bytes = frame_file.read_bytes()
        if not frame_bytes:
            raise ValueError(f"Frame file is empty: {frame_file}")
        if frame_file.name.startswith("positive_"):
            positive_frames.append(frame_bytes)
            continue
        if frame_file.name.startswith("negative_"):
            try:
                neg_idx = int(frame_file.stem.split("_")[1])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Unsupported negative frame naming: {frame_file.name}") from exc
            negative_frames.setdefault(neg_idx, []).append(frame_bytes)

    if not positive_frames:
        raise FileNotFoundError(f"No positive frames found under {clip_dir}")

    candidate_videos = [positive_frames]
    for neg_idx in sorted(negative_frames):
        candidate_videos.append(negative_frames[neg_idx])
    return candidate_videos, [0]


def _load_video_mret_candidates(
    frame_root: Path,
    row: dict[str, Any],
) -> tuple[list[list[bytes]], list[int]]:
    """Load one local candidate set for one moment-retrieval query.

    MMEB-V2 currently exposes two extracted layouts:

    1. directory layout:
       ``positive_clip/`` plus ``negative_clip_<n>/``
    2. flat frame layout:
       ``positive_*.jpg`` plus ``negative_<n>_*.jpg``
    """
    clips_dir_path = str(row.get("clips_dir_path", "") or "")
    if not clips_dir_path:
        raise ValueError("Missing clips_dir_path for video_mret query")
    clip_name = Path(clips_dir_path).name
    clip_dir = frame_root / clip_name
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")

    if (clip_dir / "positive_clip").exists():
        return _load_video_mret_directory_candidates(clip_dir)
    return _load_video_mret_flat_candidates(clip_dir)


def _momentseeker_clip_name(output_path: str) -> str:
    """Derive the extracted clip directory name from a raw output path."""
    return output_path.replace("/", "_").split(".mp4")[0]


def _load_momentseeker_query_media(frame_root: Path, row: dict[str, Any]) -> tuple[str, list[bytes] | bytes | None]:
    """Load the query media referenced by one MomentSeeker row."""
    input_frames = str(row.get("input_frames", "") or "")
    if not input_frames:
        return "", None

    input_path = Path(input_frames)
    if input_path.suffix.lower() == ".mp4":
        video_name = _momentseeker_clip_name(input_frames)
        query_dir = frame_root / "video_frames" / video_name
        return "video", list(_load_frame_dir_bytes_cached(str(query_dir)))

    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image_path = frame_root / "query_images" / input_path.parent.name / input_path.name
        if not image_path.exists():
            raise FileNotFoundError(f"MomentSeeker query image not found: {image_path}")
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise ValueError(f"MomentSeeker query image is empty: {image_path}")
        return "image", image_bytes

    raise ValueError(f"Unsupported MomentSeeker input_frames value: {input_frames!r}")


def _build_momentseeker_candidate_entry(
    entry: dict[str, Any],
    *,
    candidate_id: int,
    frame_root: Path,
) -> dict[str, Any]:
    """Build one converted candidate row from one MomentSeeker entry."""
    output_path = str((entry or {}).get("output_path", "") or "")
    if not output_path:
        raise ValueError("Missing output_path in MomentSeeker candidate entry")

    clip_name = _momentseeker_clip_name(output_path)
    clip_dir = frame_root / "video_frames" / clip_name
    candidate = dict(entry)
    candidate["id"] = candidate_id
    candidate["video"] = list(_load_frame_dir_bytes_cached(str(clip_dir)))
    return candidate


def convert_video_momentseeker_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> Path:
    """Convert the special MomentSeeker dataset into rerun layout."""

    if get_pipeline(dataset_name) != "video_momentseeker":
        raise ValueError(f"{dataset_name!r} is not a video_momentseeker dataset")

    output_path = ensure_output_root(output_root / "video-tasks" / dataset_name)
    frame_root = _resolve_frame_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    records = _load_annotation_records(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )

    def _build_query_payload(entry: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        row_index, row = entry
        query = dict(row)
        query["id"] = row_index
        # Keep both keys present in the stored schema. Downstream runtime uses
        # the explicit `image=None` / `video=None` combination to distinguish
        # text-only, image-conditioned, and video-conditioned queries without
        # relying on missing-field heuristics.
        query["image"] = None
        query["video"] = None
        media_kind, media_value = _load_momentseeker_query_media(frame_root, row)
        if media_kind:
            query[media_kind] = media_value
        positive_entries = [dict(item) for item in _ensure_plain_list(row.get("positive_frames"))]
        negative_entries = [dict(item) for item in _ensure_plain_list(row.get("negative_frames"))]
        return query, positive_entries, negative_entries

    payloads = list(
        _parallel_map_in_order(
            list(enumerate(records)),
            _build_query_payload,
            max_workers=max_workers,
            desc=f"{dataset_name}: reading clips",
        )
    )

    queries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    candidate_id_counter = 0

    for query, positive_entries, negative_entries in payloads:
        query_id = int(query["id"])
        queries.append(query)

        query_candidate_ids: list[int] = []
        candidate_scores: list[float] = []

        if not positive_entries and not negative_entries:
            raise ValueError(
                f"[{dataset_name}] query {query_id} has no positive or negative candidates"
            )

        for entry in positive_entries:
            candidate_id = candidate_id_counter
            candidate_id_counter += 1
            candidates.append(
                _build_momentseeker_candidate_entry(
                    entry,
                    candidate_id=candidate_id,
                    frame_root=frame_root,
                )
            )
            query_candidate_ids.append(candidate_id)
            candidate_scores.append(1.0)

        for entry in negative_entries:
            candidate_id = candidate_id_counter
            candidate_id_counter += 1
            candidates.append(
                _build_momentseeker_candidate_entry(
                    entry,
                    candidate_id=candidate_id,
                    frame_root=frame_root,
                )
            )
            query_candidate_ids.append(candidate_id)
            candidate_scores.append(0.0)

        qrels.append(
            {
                "query_id": query_id,
                "mode": "exhaustive",
                "candidate_ids": query_candidate_ids,
                "candidate_scores": candidate_scores,
            }
        )

    _write_lance(queries, output_path / "queries.lance")
    _write_lance(candidates, output_path / "candidates.lance")
    _write_lance(qrels, output_path / "qrels.lance")
    _write_metadata(output_path, dataset_name=dataset_name)
    return output_path


def convert_video_mret_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> Path:
    """Convert one MMEB-V2 video moment retrieval dataset into rerun layout."""

    if get_pipeline(dataset_name) != "video_moment_retrieval":
        raise ValueError(f"{dataset_name!r} is not a video_moment_retrieval dataset")

    output_path = ensure_output_root(output_root / "video-tasks" / dataset_name)
    frame_root = _resolve_frame_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    records = _load_annotation_records(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )

    def _build_query_payload(entry: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any], list[list[bytes]], list[int]]:
        row_index, row = entry
        query = dict(row)
        query["id"] = row_index
        query["video"] = _load_video_mret_query_frames(frame_root, row)
        candidate_videos, positive_indices = _load_video_mret_candidates(frame_root, row)
        return query, candidate_videos, positive_indices

    payloads = list(
        _parallel_map_in_order(
            list(enumerate(records)),
            _build_query_payload,
            max_workers=max_workers,
            desc=f"{dataset_name}: reading clips",
        )
    )

    queries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    candidate_id_counter = 0

    for query, candidate_videos, positive_indices in payloads:
        query_id = int(query["id"])
        queries.append(query)

        query_candidate_ids: list[int] = []
        for candidate_video in candidate_videos:
            candidate_id = candidate_id_counter
            candidate_id_counter += 1
            candidates.append(
                {
                    "id": candidate_id,
                    "video": candidate_video,
                }
            )
            query_candidate_ids.append(candidate_id)

        qrels.append(
            {
                "query_id": query_id,
                "mode": "exhaustive",
                "candidate_ids": list(query_candidate_ids),
                "candidate_scores": [
                    1.0 if option_index in positive_indices else 0.0
                    for option_index in range(len(query_candidate_ids))
                ],
            }
        )

    _write_lance(queries, output_path / "queries.lance")
    _write_lance(candidates, output_path / "candidates.lance")
    _write_lance(qrels, output_path / "qrels.lance")
    _write_metadata(output_path, dataset_name=dataset_name)
    return output_path


def convert_all_video_cls_datasets(
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> list[Path]:
    """Convert all video classification datasets declared in the dataset list."""
    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("video_cls", modality=VIDEO_MODALITY):
        outputs.append(
            convert_video_cls_dataset(
                spec.name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_video_qa_datasets(
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> list[Path]:
    """Convert all video question-answer datasets declared in the dataset list."""
    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("video_qa", modality=VIDEO_MODALITY):
        outputs.append(
            convert_video_qa_dataset(
                spec.name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_video_ret_datasets(
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> list[Path]:
    """Convert all video text-retrieval datasets declared in the dataset list."""
    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("video_text_retrieval", modality=VIDEO_MODALITY):
        outputs.append(
            convert_video_ret_dataset(
                spec.name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_video_mret_datasets(
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> list[Path]:
    """Convert all video moment-retrieval datasets declared in the dataset list."""
    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("video_moment_retrieval", modality=VIDEO_MODALITY):
        outputs.append(
            convert_video_mret_dataset(
                spec.name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_video_momentseeker_datasets(
    output_root: Path,
    *,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
) -> list[Path]:
    """Convert all MomentSeeker-style datasets declared in the dataset list."""
    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("video_momentseeker", modality=VIDEO_MODALITY):
        outputs.append(
            convert_video_momentseeker_dataset(
                spec.name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
    )
    return outputs
