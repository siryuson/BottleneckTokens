"""Convert MMEB-V2 image evaluation datasets into Lance retrieval artifacts.

This module covers image-side pipelines whose queries may contain text, images,
or both. It reads local MMEB-eval annotations plus MMEB-V2 media assets and
writes ``queries``, ``candidates``, ``qrels``, and ``metadata`` outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, TypeVar

import lance
import pyarrow as pa
from tqdm.auto import tqdm

from .bindings import get_source_by_role, read_parquet_shards, resolve_source_path
from .common import ensure_output_root, get_metadata_task_type, write_minimal_metadata
from .datasets import (
    IMAGE_MODALITY,
    get_modality,
    get_pipeline,
    get_spec,
    get_specs_by_modality,
    get_specs_by_pipeline,
)

CandidateKey = TypeVar("CandidateKey", bound=Hashable)
QrelsModeStrategy = Literal["sparse", "auto"]

QUERY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("qry_text", pa.string()),
        pa.field("qry_inst", pa.string()),
        pa.field("qry_img_path", pa.string()),
        pa.field("image", pa.binary()),
    ]
)

TEXT_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)

IMAGE_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("tgt_text", pa.string()),
        pa.field("tgt_inst", pa.string()),
        pa.field("tgt_img_path", pa.string(), nullable=False),
        pa.field("image", pa.binary()),
    ]
)

QRELS_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
        pa.field("candidate_ids", pa.list_(pa.int64())),
        pa.field("candidate_scores", pa.list_(pa.float32())),
    ]
)


def _read_image_bytes(path: Path) -> bytes:
    """Read one required image file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path.read_bytes()


def _write_lance(records: list[dict[str, Any]], path: Path, schema: pa.Schema) -> None:
    """Write one typed record list into a Lance dataset."""
    table = pa.Table.from_pylist(records, schema=schema)
    lance.write_dataset(table, str(path), mode="create")


def _is_blank_text_candidate(text: str | None) -> bool:
    """Return whether one text candidate should be treated as blank."""
    if text is None:
        return True
    return not text.strip()


def _dedupe_preserve_order(values: Sequence[CandidateKey]) -> list[CandidateKey]:
    """Deduplicate a sequence without changing first-seen order."""
    deduped: list[CandidateKey] = []
    seen: set[CandidateKey] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _validate_text_candidate_groups(
    dataset_name: str,
    candidate_groups: list[list[str | None]],
) -> list[list[str]]:
    """Validate and normalize per-query text candidate groups."""
    validated_groups: list[list[str]] = []

    for row_index, candidates in enumerate(candidate_groups):
        if dataset_name == "OK-VQA":
            if candidates and _is_blank_text_candidate(candidates[0]):
                raise ValueError(
                    f"{dataset_name} row {row_index} loses the positive text candidate after blank filtering"
                )
            filtered_candidates = [
                str(candidate)
                for candidate in candidates
                if not _is_blank_text_candidate(candidate)
            ]
            if not filtered_candidates:
                raise ValueError(
                    f"{dataset_name} row {row_index} has no non-blank text candidates after blank filtering"
                )
            validated_groups.append(filtered_candidates)
            continue

        if any(_is_blank_text_candidate(candidate) for candidate in candidates):
            raise ValueError(
                f"{dataset_name} row {row_index} contains blank text candidates"
            )
        validated_groups.append([str(candidate) for candidate in candidates])

    return validated_groups


def _parallel_read_images(
    image_paths: Sequence[Path],
    *,
    max_workers: int | None = None,
    desc: str | None = None,
) -> list[bytes]:
    """Read required images in parallel while preserving input order."""
    results: list[bytes | None] = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_read_image_bytes, path): index
            for index, path in enumerate(image_paths)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=desc or "Reading images",
        ):
            index = future_to_index[future]
            results[index] = future.result()
    if any(result is None for result in results):
        raise RuntimeError("parallel image loading produced incomplete results")
    return [result for result in results if result is not None]


def _parallel_read_optional_images(
    image_paths: Sequence[Path | None],
    *,
    max_workers: int | None = None,
    desc: str | None = None,
) -> list[bytes | None]:
    """Read optional images in parallel while preserving input order."""
    results: list[bytes | None] = [None] * len(image_paths)
    indexed_paths = [(index, path) for index, path in enumerate(image_paths) if path is not None]
    if not indexed_paths:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_read_image_bytes, path): index
            for index, path in indexed_paths
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=desc or "Reading images",
        ):
            index = future_to_index[future]
            results[index] = future.result()
    for index, path in indexed_paths:
        if results[index] is None:
            raise RuntimeError(f"parallel image loading produced no result for {path}")
    return results


def _resolve_image_annotation_dir(
    dataset_name: str,
    *,
    mmeb_eval_root: Path | None,
    source_overrides: dict[str, Path] | None,
) -> Path:
    """Resolve the local annotation directory for one image dataset."""
    spec = get_spec(dataset_name)
    annotation_source = get_source_by_role(spec, "annotation")
    if annotation_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing annotation source in dataset list")
    return resolve_source_path(
        annotation_source,
        mmeb_eval_root=mmeb_eval_root,
        source_overrides=source_overrides,
    )


def _resolve_image_media_root(
    dataset_name: str,
    *,
    mmeb_v2_root: Path | None,
    source_overrides: dict[str, Path] | None,
) -> Path:
    """Resolve the local MMEB-V2 media root for one image dataset."""
    spec = get_spec(dataset_name)
    media_source = get_source_by_role(spec, "media_root")
    if media_source is None:
        raise FileNotFoundError(f"[{dataset_name}] Missing media_root source in dataset list")
    return resolve_source_path(
        media_source,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )


def _build_text_candidates_first_seen(
    candidate_groups: list[list[str]],
) -> tuple[dict[str, int], list[dict[str, Any]], list[str]]:
    """Build a first-seen global text candidate table from local groups."""
    candidate_id_by_text: dict[str, int] = {}
    candidate_records: list[dict[str, Any]] = []
    ordered_candidates: list[str] = []

    for candidates in candidate_groups:
        for candidate in candidates:
            if candidate not in candidate_id_by_text:
                candidate_id = len(candidate_records)
                candidate_id_by_text[candidate] = candidate_id
                ordered_candidates.append(candidate)
                candidate_records.append(
                    {
                        "id": candidate_id,
                        "text": candidate,
                    }
                )

    return candidate_id_by_text, candidate_records, ordered_candidates


def _build_image_candidates_first_seen(
    candidate_groups: list[list[tuple[CandidateKey, str, str | None]]],
    *,
    max_workers: int | None,
    desc: str,
    image_root: Path,
) -> tuple[dict[CandidateKey, int], list[dict[str, Any]], list[CandidateKey]]:
    """Build a first-seen global image candidate table from local groups."""
    candidate_id_by_key: dict[CandidateKey, int] = {}
    ordered_specs: list[tuple[CandidateKey, str, str | None]] = []

    for group in candidate_groups:
        for key, img_path, text in group:
            if key in candidate_id_by_key:
                continue
            candidate_id_by_key[key] = len(ordered_specs)
            ordered_specs.append((key, img_path, text))

    image_bytes = _parallel_read_images(
        [image_root / img_path for _, img_path, _ in ordered_specs],
        max_workers=max_workers,
        desc=desc,
    )

    candidate_records: list[dict[str, Any]] = []
    ordered_keys: list[CandidateKey] = []
    for candidate_id, ((key, img_path, text), image_blob) in enumerate(
        zip(ordered_specs, image_bytes, strict=True)
    ):
        ordered_keys.append(key)
        candidate_records.append(
            {
                "id": candidate_id,
                "tgt_text": text,
                "tgt_inst": None,
                "tgt_img_path": img_path,
                "image": image_blob,
            }
        )

    return candidate_id_by_key, candidate_records, ordered_keys


def _build_query_records(
    *,
    qry_texts: list[str | None],
    qry_insts: list[str | None],
    qry_img_paths: list[str | None],
    image_bytes: list[bytes | None],
    desc: str,
) -> list[dict[str, Any]]:
    """Materialize query rows from normalized query text, images, and ids."""
    queries: list[dict[str, Any]] = []
    for query_id, (qry_text, qry_inst, qry_img_path, image_blob) in enumerate(
        tqdm(
            zip(qry_texts, qry_insts, qry_img_paths, image_bytes, strict=True),
            total=len(qry_texts),
            desc=desc,
        )
    ):
        queries.append(
            {
                "id": query_id,
                "qry_text": qry_text,
                "qry_inst": qry_inst,
                "qry_img_path": qry_img_path,
                "image": image_blob,
            }
        )
    return queries


def _build_sparse_qrels(
    candidate_groups: list[list[CandidateKey]],
    candidate_id_by_key: dict[CandidateKey, int],
) -> list[dict[str, Any]]:
    """Build global sparse qrels for text/image candidate groups."""
    qrels: list[dict[str, Any]] = []
    for query_id, candidates in enumerate(candidate_groups):
        local_candidates = _dedupe_preserve_order(candidates)
        qrels.append(
            {
                "query_id": query_id,
                "mode": "sparse",
                "candidate_ids": [candidate_id_by_key[local_candidates[0]]],
                "candidate_scores": [1.0],
            }
        )
    return qrels


def _build_conditional_qrels(
    candidate_groups: list[list[CandidateKey]],
    candidate_id_by_key: dict[CandidateKey, int],
    *,
    global_candidates: Sequence[CandidateKey],
) -> list[dict[str, Any]]:
    """Infer per-query qrels mode from local-vs-global candidate coverage.

    - use ``sparse`` when one query is evaluated against the full global pool
    - use ``exhaustive`` when one query has its own local candidate subset
    """
    qrels: list[dict[str, Any]] = []
    global_candidate_set = set(global_candidates)

    for query_id, candidates in enumerate(candidate_groups):
        local_candidates = _dedupe_preserve_order(candidates)
        if set(local_candidates) == global_candidate_set:
            qrels.append(
                {
                    "query_id": query_id,
                    "mode": "sparse",
                    "candidate_ids": [candidate_id_by_key[local_candidates[0]]],
                    "candidate_scores": [1.0],
                }
            )
            continue

        qrels.append(
            {
                "query_id": query_id,
                "mode": "exhaustive",
                "candidate_ids": [candidate_id_by_key[candidate] for candidate in local_candidates],
                "candidate_scores": [1.0] + [0.0] * (len(local_candidates) - 1),
            }
        )
    return qrels


def _convert_image_text_candidate_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    qrels_strategy: QrelsModeStrategy,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one image dataset whose candidates are text labels or captions."""
    annotation_dir = _resolve_image_annotation_dir(
        dataset_name,
        mmeb_eval_root=mmeb_eval_root,
        source_overrides=source_overrides,
    )
    image_root = _resolve_image_media_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    output_path = ensure_output_root(output_root / "image-tasks" / dataset_name)

    table = read_parquet_shards(annotation_dir)
    qry_texts = table["qry_text"].to_pylist()
    qry_img_paths = table["qry_img_path"].to_pylist()
    qry_insts = table["qry_inst"].to_pylist() if "qry_inst" in table.column_names else [None] * table.num_rows
    tgt_texts = table["tgt_text"].to_pylist()
    tgt_texts = _validate_text_candidate_groups(dataset_name, tgt_texts)

    image_bytes = _parallel_read_optional_images(
        [image_root / path if path else None for path in qry_img_paths],
        max_workers=max_workers,
        desc=f"{dataset_name}: reading query images",
    )

    candidate_id_by_text, candidate_records, ordered_candidates = _build_text_candidates_first_seen(tgt_texts)
    queries = _build_query_records(
        qry_texts=qry_texts,
        qry_insts=qry_insts,
        qry_img_paths=qry_img_paths,
        image_bytes=image_bytes,
        desc=f"{dataset_name}: building query rows",
    )
    if qrels_strategy == "sparse":
        qrels = _build_sparse_qrels(tgt_texts, candidate_id_by_text)
    else:
        qrels = _build_conditional_qrels(
            tgt_texts,
            candidate_id_by_text,
            global_candidates=ordered_candidates,
        )

    _write_lance(queries, output_path / "queries.lance", QUERY_SCHEMA)
    _write_lance(candidate_records, output_path / "candidates.lance", TEXT_CANDIDATE_SCHEMA)
    _write_lance(qrels, output_path / "qrels.lance", QRELS_SCHEMA)
    write_minimal_metadata(
        output_path,
        dataset_name=dataset_name,
        task_type=get_metadata_task_type(dataset_name, get_pipeline(dataset_name)),
        modality=get_modality(dataset_name),
    )
    return output_path


def _convert_image_media_candidate_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    candidate_key_builder: Callable[[str, str | None], CandidateKey],
    query_has_image: bool,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one image dataset whose candidates are image assets."""
    annotation_dir = _resolve_image_annotation_dir(
        dataset_name,
        mmeb_eval_root=mmeb_eval_root,
        source_overrides=source_overrides,
    )
    image_root = _resolve_image_media_root(
        dataset_name,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
    )
    output_path = ensure_output_root(output_root / "image-tasks" / dataset_name)

    table = read_parquet_shards(annotation_dir)
    qry_texts = table["qry_text"].to_pylist()
    qry_img_paths = table["qry_img_path"].to_pylist() if "qry_img_path" in table.column_names else [""] * table.num_rows
    qry_insts = table["qry_inst"].to_pylist() if "qry_inst" in table.column_names else [None] * table.num_rows
    tgt_text_groups = table["tgt_text"].to_pylist()
    tgt_img_path_groups = table["tgt_img_path"].to_pylist()

    query_image_bytes = _parallel_read_optional_images(
        [image_root / path if query_has_image and path else None for path in qry_img_paths],
        max_workers=max_workers,
        desc=f"{dataset_name}: reading query images",
    )

    candidate_groups: list[list[tuple[CandidateKey, str, str | None]]] = []
    candidate_key_groups: list[list[CandidateKey]] = []
    for tgt_img_paths, tgt_texts in zip(tgt_img_path_groups, tgt_text_groups, strict=True):
        group: list[tuple[CandidateKey, str, str | None]] = []
        key_group: list[CandidateKey] = []
        for tgt_img_path, tgt_text in zip(tgt_img_paths, tgt_texts, strict=True):
            key = candidate_key_builder(tgt_img_path, tgt_text)
            group.append((key, tgt_img_path, tgt_text))
            key_group.append(key)
        candidate_groups.append(group)
        candidate_key_groups.append(key_group)

    candidate_id_by_key, candidate_records, ordered_keys = _build_image_candidates_first_seen(
        candidate_groups,
        max_workers=max_workers,
        desc=f"{dataset_name}: reading candidate images",
        image_root=image_root,
    )
    queries = _build_query_records(
        qry_texts=qry_texts,
        qry_insts=qry_insts,
        qry_img_paths=qry_img_paths,
        image_bytes=query_image_bytes,
        desc=f"{dataset_name}: building query rows",
    )
    qrels = _build_conditional_qrels(
        candidate_key_groups,
        candidate_id_by_key,
        global_candidates=ordered_keys,
    )

    _write_lance(queries, output_path / "queries.lance", QUERY_SCHEMA)
    _write_lance(candidate_records, output_path / "candidates.lance", IMAGE_CANDIDATE_SCHEMA)
    _write_lance(qrels, output_path / "qrels.lance", QRELS_SCHEMA)
    write_minimal_metadata(
        output_path,
        dataset_name=dataset_name,
        task_type=get_metadata_task_type(dataset_name, get_pipeline(dataset_name)),
        modality=get_modality(dataset_name),
    )
    return output_path


def _candidate_key_by_image_path(img_path: str, text: str | None) -> str:
    """Deduplicate by image path only when text should not split candidates."""
    del text
    return img_path


def _candidate_key_by_image_and_text(img_path: str, text: str | None) -> tuple[str, str]:
    """Deduplicate by both image path and paired text when both are semantic."""
    return img_path, (text or "").strip('"')


def convert_image_cls_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one MMEB-V2 image classification dataset into rerun layout."""

    if get_pipeline(dataset_name) != "image_cls":
        raise ValueError(f"{dataset_name!r} is not an image_cls dataset")

    return _convert_image_text_candidate_dataset(
        dataset_name,
        output_root,
        qrels_strategy="sparse",
        mmeb_eval_root=mmeb_eval_root,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
        max_workers=max_workers,
    )


def convert_image_qa_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one MMEB-V2 image QA dataset into rerun layout."""

    if get_pipeline(dataset_name) != "image_qa":
        raise ValueError(f"{dataset_name!r} is not an image_qa dataset")

    return _convert_image_text_candidate_dataset(
        dataset_name,
        output_root,
        qrels_strategy="auto",
        mmeb_eval_root=mmeb_eval_root,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
        max_workers=max_workers,
    )


def convert_image_i2t_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one MMEB-V2 image i2t dataset into rerun layout."""

    if get_pipeline(dataset_name) != "image_i2t":
        raise ValueError(f"{dataset_name!r} is not an image_i2t dataset")

    return _convert_image_text_candidate_dataset(
        dataset_name,
        output_root,
        qrels_strategy="auto",
        mmeb_eval_root=mmeb_eval_root,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
        max_workers=max_workers,
    )


def convert_image_t2i_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one MMEB-V2 image t2i dataset into rerun layout."""

    if get_pipeline(dataset_name) != "image_t2i":
        raise ValueError(f"{dataset_name!r} is not an image_t2i dataset")

    return _convert_image_media_candidate_dataset(
        dataset_name,
        output_root,
        candidate_key_builder=_candidate_key_by_image_path,
        query_has_image=False,
        mmeb_eval_root=mmeb_eval_root,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
        max_workers=max_workers,
    )


def convert_image_i2i_vg_dataset(
    dataset_name: str,
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Convert one MMEB-V2 image i2i/vg dataset into rerun layout."""

    if get_pipeline(dataset_name) != "image_i2i_vg":
        raise ValueError(f"{dataset_name!r} is not an image_i2i_vg dataset")

    return _convert_image_media_candidate_dataset(
        dataset_name,
        output_root,
        candidate_key_builder=_candidate_key_by_image_and_text,
        query_has_image=True,
        mmeb_eval_root=mmeb_eval_root,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
        max_workers=max_workers,
    )


def convert_all_image_cls_datasets(
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Convert all MMEB-V2 image classification datasets into rerun layout."""

    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("image_cls", modality=IMAGE_MODALITY):
        outputs.append(
            convert_image_cls_dataset(
                spec.name,
                output_root,
                mmeb_eval_root=mmeb_eval_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_image_qa_datasets(
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Convert all MMEB-V2 image QA datasets into rerun layout."""

    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("image_qa", modality=IMAGE_MODALITY):
        outputs.append(
            convert_image_qa_dataset(
                spec.name,
                output_root,
                mmeb_eval_root=mmeb_eval_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_image_i2t_datasets(
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Convert all MMEB-V2 image i2t datasets into rerun layout."""

    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("image_i2t", modality=IMAGE_MODALITY):
        outputs.append(
            convert_image_i2t_dataset(
                spec.name,
                output_root,
                mmeb_eval_root=mmeb_eval_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_image_t2i_datasets(
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Convert all MMEB-V2 image t2i datasets into rerun layout."""

    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("image_t2i", modality=IMAGE_MODALITY):
        outputs.append(
            convert_image_t2i_dataset(
                spec.name,
                output_root,
                mmeb_eval_root=mmeb_eval_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_image_i2i_vg_datasets(
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Convert all MMEB-V2 image i2i/vg datasets into rerun layout."""

    outputs: list[Path] = []
    for spec in get_specs_by_pipeline("image_i2i_vg", modality=IMAGE_MODALITY):
        outputs.append(
            convert_image_i2i_vg_dataset(
                spec.name,
                output_root,
                mmeb_eval_root=mmeb_eval_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_all_image_datasets(
    output_root: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Convert all MMEB-V2 image datasets into rerun layout."""

    outputs: list[Path] = []
    pipeline_fns = _get_image_pipeline_fns()
    for spec in get_specs_by_modality(IMAGE_MODALITY):
        pipeline_fn = pipeline_fns.get(spec.pipeline)
        if pipeline_fn is None:
            raise NotImplementedError(
                f"MMEB-V2 image conversion is not implemented yet for {spec.name!r} "
                f"(pipeline={spec.pipeline})"
            )
        outputs.append(
            pipeline_fn(
                spec.name,
                output_root,
                mmeb_eval_root=mmeb_eval_root,
                mmeb_v2_root=mmeb_v2_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def convert_imagenet_1k(
    output_dir: Path,
    *,
    mmeb_eval_root: Path | None = None,
    mmeb_v2_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Run the ImageNet-1K rerun conversion."""

    return convert_image_cls_dataset(
        "ImageNet-1K",
        output_dir,
        mmeb_eval_root=mmeb_eval_root,
        mmeb_v2_root=mmeb_v2_root,
        source_overrides=source_overrides,
        max_workers=max_workers,
    )

def _get_image_pipeline_fns():
    """Return the image pipeline-to-function map used by bulk helpers."""
    return {
        "image_cls": convert_image_cls_dataset,
        "image_qa": convert_image_qa_dataset,
        "image_i2t": convert_image_i2t_dataset,
        "image_t2i": convert_image_t2i_dataset,
        "image_i2i_vg": convert_image_i2i_vg_dataset,
    }
