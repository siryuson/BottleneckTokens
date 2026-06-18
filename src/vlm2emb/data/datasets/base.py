"""Base Lance dataset classes for the BToks public runtime.

This module now centers the runtime data model around one unified sample
dataset abstraction:

1. ``LanceDataset`` keeps exact raw-table semantics.
2. ``SampleLanceDataset`` treats one Lance table as the primary iteration
   domain, optionally enriches each primary row through one or more
   side-table extensions, and finally applies exactly one runtime transform.

The intent is to make single-table and multi-table datasets follow the same
execution model:

``primary rows -> extension flattening -> transform(record) -> item``

Legacy direct/eval aliases remain at the bottom for import compatibility, but
new training datasets should use ``SampleLanceDataset`` directly unless a
future train-specific abstraction gains real behavior.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import lance as lance_lib
import numpy as np
from torch.utils.data import Dataset

from vlm2emb.data.datasets.const import decode_image, validate_runtime_sample

SampleTransform = Callable[[dict[str, Any]], dict[str, Any]]
ExtensionKey = str | Callable[[dict[str, Any]], Any]
logger = logging.getLogger(__name__)


def _merge_sample_metadata(
    sample: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge explicit runtime metadata into one sample."""
    if not metadata and "metadata" not in sample:
        return sample

    sample_metadata = sample.get("metadata")
    if sample_metadata is None:
        merged_metadata: dict[str, Any] = {}
    elif isinstance(sample_metadata, dict):
        merged_metadata = dict(sample_metadata)
    else:
        raise TypeError(
            f"{type(sample).__name__}.metadata must be dict when present, "
            f"got {type(sample_metadata).__name__}"
        )

    merged_metadata = dict(metadata)
    merged_metadata.update(sample_metadata or {})
    return {**sample, "metadata": merged_metadata}


def _normalize_read_columns(
    read_columns: list[str] | tuple[str, ...] | None,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> list[str] | None:
    """Normalize raw/read column aliases into one ordered unique list."""
    if read_columns is not None and columns is not None:
        raise ValueError("Pass only one of read_columns or columns.")
    effective = read_columns if read_columns is not None else columns
    if effective is None:
        return None
    normalized: list[str] = []
    for name in effective:
        if not isinstance(name, str) or not name:
            raise TypeError(f"read column names must be non-empty strings, got {name!r}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def _sql_literal(value: Any) -> str:
    """Render one scalar as a DataFusion-compatible SQL literal."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"Non-finite float keys are not supported in Lance filters: {value!r}")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"Unsupported keyed lookup value type for Lance filters: {type(value).__name__}")


def _rename_extension_row(
    row: dict[str, Any],
    *,
    extension_index: int,
    column_name_map: dict[str, str],
) -> dict[str, Any]:
    """Rename one extension row while guarding against post-map duplicates."""
    renamed: dict[str, Any] = {}
    for source_name, value in row.items():
        target_name = column_name_map.get(source_name, source_name)
        if target_name in renamed:
            raise ValueError(
                f"Extension #{extension_index} maps multiple source columns to '{target_name}'."
            )
        renamed[target_name] = value
    return renamed


def _merge_extension_row(
    record: dict[str, Any],
    extension_row: dict[str, Any],
    *,
    extension_index: int,
) -> dict[str, Any]:
    """Flatten one extension row into one record with strict collision checks."""
    conflicts = sorted(set(record) & set(extension_row))
    if conflicts:
        raise ValueError(
            f"Extension #{extension_index} would overwrite existing fields: {conflicts}"
        )
    return {**record, **extension_row}


def _is_empty_extension_key(key: Any) -> bool:
    """Return whether an extension key intentionally has no lookup target."""
    if key is None:
        return True
    if isinstance(key, str) and not key.strip():
        return True
    return False


def _decode_eval_record(record: dict[str, Any]) -> dict[str, Any]:
    """Decode raw eval media bytes into PIL objects."""
    decoded = dict(record)

    image_payload = decoded.get("image")
    if isinstance(image_payload, bytes):
        decoded["image"] = decode_image(image_payload)

    images = decoded.get("images")
    if isinstance(images, list):
        decoded["images"] = [
            decode_image(image) if isinstance(image, bytes) else image
            for image in images
        ]

    video_payload = decoded.get("video")
    if isinstance(video_payload, list):
        decoded["video"] = [
            decode_image(frame) if isinstance(frame, bytes) else frame
            for frame in video_payload
        ]

    if "images" not in decoded:
        single_image = decoded.get("image")
        decoded["images"] = [] if single_image is None else [single_image]
    elif decoded["images"] is None:
        decoded["images"] = []
    return decoded


def _apply_transform_chain(
    record: dict[str, Any],
    *,
    transforms: tuple[SampleTransform, ...],
) -> dict[str, Any]:
    """Apply one ordered transform chain using a pickle-safe top-level callable.

    DataLoader worker startup under ``spawn`` needs every callable reachable from
    the dataset object to be pickleable. Keeping the composition logic in this
    top-level helper avoids local closures inside dataset constructors.
    """
    result = record
    for transform in transforms:
        result = transform(result)
    return result


def _sample_video_eval_record(
    record: dict[str, Any],
    *,
    num_frames: int,
) -> dict[str, Any]:
    """Sample eval frames after media decoding."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive for video eval sampling.")

    sampled = dict(record)
    images = sampled.get("images")
    if images is None and "video" in sampled:
        video_payload = sampled.get("video")
        if isinstance(video_payload, list):
            images = video_payload
            sampled["images"] = images
    images = images or []
    if not images:
        sampled.setdefault("images", [])
        return sampled

    if len(images) > num_frames:
        sampled_indices = np.linspace(0, len(images) - 1, num_frames, dtype=int).tolist()
        sampled["images"] = [images[index] for index in sampled_indices]
    else:
        sampled_indices = list(range(len(images)))

    media_metadata = sampled.get("media_metadata")
    if isinstance(media_metadata, dict):
        merged_metadata = dict(media_metadata)
    else:
        merged_metadata = {}
    merged_metadata.setdefault("total_num_frames", len(images))
    merged_metadata["sampled_indices"] = sampled_indices
    sampled["media_metadata"] = merged_metadata
    return sampled


def build_video_eval_transform(*, num_frames: int = 8) -> SampleTransform:
    """Build one pickle-safe eval transform that keeps uniformly sampled frames only."""
    return partial(_sample_video_eval_record, num_frames=num_frames)


@dataclass(frozen=True)
class ResolvedTrainDatasetSpec:
    """Resolved construction facts for one training dataset instance."""

    dataset_type: str
    dataset_name: str | None
    primary_path: str
    lookup_paths: dict[str, str] = field(default_factory=dict)
    split: str | None = None
    subset: str | None = None
    variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedRetrievalEvalSpec:
    """Resolved construction facts for one retrieval eval dataset instance."""

    dataset_name: str | None
    task_type: str | None
    queries_path: str
    candidates_path: str
    qrels_path: str
    split: str | None = None
    subset: str | None = None
    variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LanceTableExtension:
    """One side-table lookup to flatten into the primary record stream.

    Notes:
        - ``lance_path`` always points to the side table.
        - ``key`` is resolved from the current flattened record, so later
          extensions may depend on fields injected by earlier extensions.
        - ``column_name is None`` means positional lookup by row index.
        - ``column_name`` means keyed lookup with unique-row semantics.
        - ``column_name_map`` only renames fetched side-table fields before
          flattening. It never renames the lookup key itself.
    """

    lance_path: str
    key: ExtensionKey
    read_columns: list[str]
    column_name: str | None = None
    missing: Literal["error", "skip"] = "error"
    column_name_map: dict[str, str] = field(default_factory=dict)


class LanceDataset(Dataset):
    """Exact raw Lance dataset wrapper."""

    def __init__(
        self,
        lance_path: str,
        read_columns: list[str] | None = None,
        *,
        columns: list[str] | None = None,
        lazy: bool = True,
    ) -> None:
        self._lance_path = lance_path
        self._read_columns = _normalize_read_columns(read_columns, columns=columns)
        self._columns = self._read_columns
        self._lazy = lazy
        self._ds: lance_lib.LanceDataset | None = None
        self._length: int | None = None

        if not self._lazy:
            _ = self.ds

    @property
    def ds(self) -> lance_lib.LanceDataset:
        """Lazy-load the Lance dataset handle."""
        if self._ds is None:
            logger.debug("LanceDataset.ds open start path=%s", self._lance_path)
            self._ds = lance_lib.dataset(self._lance_path)
            logger.debug("LanceDataset.ds open done path=%s", self._lance_path)
        return self._ds

    @property
    def raw_schema(self):
        """Return the underlying Lance schema."""
        return self.ds.schema

    @property
    def raw_columns(self) -> list[str]:
        """Return raw column names."""
        return list(self.ds.schema.names)

    def has_column(self, name: str) -> bool:
        """Check whether the raw table exposes one column."""
        return name in self.ds.schema.names

    def __len__(self) -> int:
        if self._length is None:
            self._length = self.ds.count_rows()
        return self._length

    def to_table(
        self,
        read_columns: list[str] | None = None,
        *,
        columns: list[str] | None = None,
        filter: str | None = None,
        limit: int | None = None,
    ):
        """Read rows as a PyArrow table using exact raw columns."""
        effective_columns = _normalize_read_columns(read_columns, columns=columns)
        logger.debug(
            "LanceDataset.to_table start path=%s read_columns=%s filter=%s limit=%s",
            self._lance_path,
            effective_columns,
            filter,
            limit,
        )
        table = self.ds.to_table(columns=effective_columns, filter=filter, limit=limit)
        logger.debug(
            "LanceDataset.to_table done path=%s rows=%d columns=%d",
            self._lance_path,
            table.num_rows,
            table.num_columns,
        )
        return table

    def take_row(
        self,
        index: int,
        read_columns: list[str] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read one raw row."""
        effective_columns = _normalize_read_columns(read_columns, columns=columns)
        return self.ds.take([index], columns=effective_columns).to_pylist()[0]

    def take_rows(
        self,
        indices: list[int],
        read_columns: list[str] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Read multiple raw rows."""
        if not indices:
            return []
        effective_columns = _normalize_read_columns(read_columns, columns=columns)
        return self.ds.take(indices, columns=effective_columns).to_pylist()

    def scan_rows(
        self,
        read_columns: list[str] | None = None,
        *,
        columns: list[str] | None = None,
        batch_size: int = 128,
        batch_readahead: int = 8,
    ) -> Iterator[list[dict[str, Any]]]:
        """Iterate raw rows in batches."""
        effective_columns = _normalize_read_columns(read_columns, columns=columns)
        scanner = self.ds.scanner(
            columns=effective_columns,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
        )
        for batch in scanner.to_batches():
            yield batch.to_pylist()

    def _read_row(self, idx: int) -> dict[str, Any]:
        return self.take_row(idx, read_columns=self._read_columns)

    def _read_rows(self, indices: list[int]) -> list[dict[str, Any]]:
        return self.take_rows(indices, read_columns=self._read_columns)

    def _iter_row_batches(self) -> Iterator[list[dict[str, Any]]]:
        yield from self.scan_rows(read_columns=self._read_columns)

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        for batch in self._iter_row_batches():
            yield from batch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self.take_row(idx, read_columns=self._read_columns)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return self.take_rows(indices, read_columns=self._read_columns)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows()

    def __repr__(self) -> str:
        return f"LanceDataset(path={self._lance_path!r}, length={len(self)})"


class LanceKeyedRowResolver:
    """Resolve rows by keyed lookup with Lance index or in-memory fallback.

    This resolver is dataset-instance scoped and intentionally keeps any
    in-memory fallback index alive for the whole dataset lifetime. That matches
    the design goal of paying the indexing cost once per dataset object rather
    than once per batch.
    """

    def __init__(self, dataset: LanceDataset, key_column: str) -> None:
        self.dataset = dataset
        self.key_column = key_column
        self._row_index: dict[Hashable, int] | None = None
        self._has_scalar_index: bool | None = None

    def has_scalar_index(self) -> bool:
        if self._has_scalar_index is None:
            indices = self.dataset.ds.describe_indices()
            self._has_scalar_index = any(self.key_column in idx.field_names for idx in indices)
        return self._has_scalar_index

    def build_row_index(self) -> dict[Hashable, int]:
        if self._row_index is not None:
            return self._row_index

        index: dict[Hashable, int] = {}
        data = self.dataset.to_table(read_columns=[self.key_column]).to_pydict()
        values = data[self.key_column]
        for row_index, value in enumerate(values):
            if not isinstance(value, Hashable):
                raise TypeError(
                    f"Key column '{self.key_column}' contains non-hashable values: {type(value).__name__}"
                )
            if value in index:
                raise ValueError(
                    f"Duplicate keyed rows detected in '{self.dataset._lance_path}' for "
                    f"{self.key_column}={value!r}"
                )
            index[value] = row_index
        self._row_index = index
        return self._row_index

    def lookup_rows(
        self,
        keys: list[Any],
        *,
        columns: list[str],
        missing: Literal["error", "skip"] = "error",
    ) -> list[dict[str, Any] | None]:
        if not keys:
            return []
        unique_keys: list[Hashable] = []
        for key in keys:
            if not isinstance(key, Hashable):
                raise TypeError(
                    f"Lookup key for column '{self.key_column}' must be hashable, got {type(key).__name__}"
                )
            if key not in unique_keys:
                unique_keys.append(key)

        row_by_key: dict[Hashable, dict[str, Any]] = {}
        missing_keys: set[Hashable] = set()

        if self.has_scalar_index():
            try:
                literals = [_sql_literal(key) for key in unique_keys]
            except TypeError:
                literals = []
            if literals:
                filter_expr = (
                    f"{self.key_column} = {literals[0]}"
                    if len(literals) == 1
                    else f"{self.key_column} IN ({', '.join(literals)})"
                )
                table = self.dataset.to_table(read_columns=[self.key_column, *columns], filter=filter_expr)
                for row in table.to_pylist():
                    raw_key = row.get(self.key_column)
                    if raw_key in row_by_key:
                        raise ValueError(
                            f"Duplicate keyed rows detected in '{self.dataset._lance_path}' for "
                            f"{self.key_column}={raw_key!r}"
                        )
                    row_by_key[raw_key] = {
                        column: row[column]
                        for column in columns
                        if column in row
                    }
                missing_keys = {key for key in unique_keys if key not in row_by_key}

        if missing_keys or not self.has_scalar_index():
            row_index = self.build_row_index()
            fetch_positions: list[tuple[Hashable, int]] = []
            for key in unique_keys:
                position = row_index.get(key)
                if position is None:
                    missing_keys.add(key)
                    continue
                fetch_positions.append((key, position))

            if fetch_positions:
                rows = self.dataset.take_rows(
                    [position for _, position in fetch_positions],
                    read_columns=columns,
                )
                for (key, _), row in zip(fetch_positions, rows, strict=True):
                    row_by_key[key] = row

        if missing == "error" and missing_keys:
            raise KeyError(
                f"Missing keyed rows in '{self.dataset._lance_path}' for "
                f"{self.key_column}: {sorted(missing_keys, key=repr)}"
            )

        return [row_by_key.get(key) for key in keys]

    def get_row(self, key: Any, *, columns: list[str]) -> dict[str, Any]:
        row = self.lookup_rows([key], columns=columns, missing="error")[0]
        if row is None:
            raise KeyError(key)
        return row

    def get_rows(self, keys: list[Any], *, columns: list[str]) -> list[dict[str, Any]]:
        rows = self.lookup_rows(keys, columns=columns, missing="error")
        return [row for row in rows if row is not None]


class SampleLanceDataset(LanceDataset):
    """Sample-oriented Lance dataset with one unified transform path.

    The primary table defines iteration order and indexing semantics. Side
    tables are optional enrichments layered on top of the primary rows.

    Execution model:
        1. Read one batch of primary rows.
        2. Resolve extensions batch by batch without degrading to per-item
           primary-table access.
        3. Flatten extension rows into the current records with strict field
           collision checks.
        4. Apply one final ``transform(record) -> item``.
        5. Validate the emitted runtime item and merge dataset-level metadata.

    Subclasses that need custom batch joins outside the current one-to-one
    extension model may override ``_resolve_records`` directly.
    """

    def __init__(
        self,
        lance_path: str,
        read_columns: list[str] | None,
        extensions: list[LanceTableExtension] | None = None,
        metadata: dict[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        columns: list[str] | None = None,
        lazy: bool = True,
    ) -> None:
        normalized_columns = _normalize_read_columns(read_columns, columns=columns)
        super().__init__(
            lance_path=lance_path,
            read_columns=normalized_columns,
            lazy=lazy,
        )
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError(f"metadata must be mapping when provided, got {type(metadata).__name__}")
        self._metadata = dict(metadata or {})
        self._transform = transform
        self._extensions = list(extensions or [])
        self._sample_columns = list(normalized_columns or [])
        self._extension_datasets: dict[int, LanceDataset] = {}
        self._extension_resolvers: dict[int, LanceKeyedRowResolver] = {}

    def _get_extension_dataset(self, extension_index: int) -> LanceDataset:
        dataset = self._extension_datasets.get(extension_index)
        if dataset is None:
            extension = self._extensions[extension_index]
            dataset = LanceDataset(extension.lance_path, lazy=self._lazy)
            self._extension_datasets[extension_index] = dataset
        return dataset

    def _get_extension_resolver(self, extension_index: int) -> LanceKeyedRowResolver:
        resolver = self._extension_resolvers.get(extension_index)
        if resolver is None:
            extension = self._extensions[extension_index]
            if extension.column_name is None:
                raise ValueError(f"Extension #{extension_index} does not declare column_name.")
            resolver = LanceKeyedRowResolver(
                self._get_extension_dataset(extension_index),
                extension.column_name,
            )
            self._extension_resolvers[extension_index] = resolver
        return resolver

    def _resolve_extension_key(
        self,
        record: dict[str, Any],
        *,
        extension: LanceTableExtension,
        extension_index: int,
    ) -> Any:
        if callable(extension.key):
            return extension.key(record)
        if extension.key not in record:
            raise KeyError(
                f"Extension #{extension_index} requires key '{extension.key}' but the field is missing."
            )
        return record[extension.key]

    def _lookup_extension_rows_by_position(
        self,
        extension_index: int,
        *,
        keys: list[Any],
        columns: list[str],
        missing: Literal["error", "skip"],
    ) -> list[dict[str, Any] | None]:
        dataset = self._get_extension_dataset(extension_index)
        valid_positions: list[int] = []
        unique_positions: list[int] = []
        invalid_positions: list[int] = []

        for key in keys:
            if type(key) is not int:
                raise TypeError(
                    f"Extension #{extension_index} positional lookup expects int keys, got {type(key).__name__}"
                )
            if key < 0 or key >= len(dataset):
                invalid_positions.append(key)
                continue
            valid_positions.append(key)
            if key not in unique_positions:
                unique_positions.append(key)

        if invalid_positions and missing == "error":
            raise IndexError(
                f"Extension #{extension_index} positional lookup out of range: {invalid_positions}"
            )

        row_by_position: dict[int, dict[str, Any]] = {}
        if unique_positions:
            rows = dataset.take_rows(unique_positions, read_columns=columns)
            row_by_position = {
                position: row
                for position, row in zip(unique_positions, rows, strict=True)
            }

        return [row_by_position.get(key) for key in keys]

    def _apply_extensions(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved_records = [dict(record) for record in records]
        for extension_index, extension in enumerate(self._extensions):
            keys = [
                self._resolve_extension_key(
                    record,
                    extension=extension,
                    extension_index=extension_index,
                )
                for record in resolved_records
            ]
            empty_key_mask = [
                extension.missing == "skip" and _is_empty_extension_key(key)
                for key in keys
            ]
            lookup_positions = [
                position
                for position, is_empty in enumerate(empty_key_mask)
                if not is_empty
            ]
            lookup_keys = [keys[position] for position in lookup_positions]
            if extension.column_name is None:
                looked_up_rows = self._lookup_extension_rows_by_position(
                    extension_index,
                    keys=lookup_keys,
                    columns=extension.read_columns,
                    missing=extension.missing,
                )
            else:
                looked_up_rows = self._get_extension_resolver(extension_index).lookup_rows(
                    lookup_keys,
                    columns=extension.read_columns,
                    missing=extension.missing,
                )
            extension_rows: list[dict[str, Any] | None] = [None] * len(keys)
            for position, extension_row in zip(lookup_positions, looked_up_rows, strict=True):
                extension_rows[position] = extension_row

            next_records: list[dict[str, Any]] = []
            for record, extension_row in zip(resolved_records, extension_rows, strict=True):
                if extension_row is None:
                    next_records.append(record)
                    continue
                renamed_row = _rename_extension_row(
                    extension_row,
                    extension_index=extension_index,
                    column_name_map=extension.column_name_map,
                )
                next_records.append(
                    _merge_extension_row(
                        record,
                        renamed_row,
                        extension_index=extension_index,
                    )
                )
            resolved_records = next_records
        return resolved_records

    def _resolve_records(
        self,
        primary_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve final flat records from primary rows.

        Subclasses may override this when they need custom batch joins that are
        outside the current one-to-one extension model. The override point is
        intentionally here, before the final runtime transform, so callers can
        still share the same validation and metadata merge path afterwards.
        """
        return self._apply_extensions(primary_rows)

    def _read_records(self, indices: list[int]) -> list[dict[str, Any]]:
        primary_rows = self.take_rows(indices, read_columns=self._read_columns)
        return self._resolve_records(primary_rows)

    def _iter_record_batches(self) -> Iterator[list[dict[str, Any]]]:
        for primary_rows in self.scan_rows(read_columns=self._read_columns):
            yield self._resolve_records(primary_rows)

    def _transform_record(self, record: dict[str, Any]) -> dict[str, Any]:
        item = self._transform(record) if self._transform is not None else dict(record)
        if not isinstance(item, dict):
            raise TypeError(
                f"{type(self).__name__}.transform must return dict, got {type(item).__name__}"
            )
        validation_overrides = item.pop("_validation", None)
        if validation_overrides is None:
            validation_overrides = {}
        if not isinstance(validation_overrides, dict):
            raise TypeError(
                f"{type(self).__name__}._validation must be dict when present, got "
                f"{type(validation_overrides).__name__}"
            )
        validated = validate_runtime_sample(
            item,
            sample_name=type(self).__name__,
            allow_visual_token_without_media=bool(
                validation_overrides.get("allow_visual_token_without_media", False)
            ),
        )
        return _merge_sample_metadata(validated, self._metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._transform_record(self._read_records([idx])[0])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self._transform_record(record) for record in self._read_records(indices)]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self._iter_record_batches():
            for record in batch:
                yield self._transform_record(record)


class EvalLanceDataset(SampleLanceDataset):
    """Sample-oriented Lance dataset for evaluation with media decoding.

    Eval datasets default to the common flattened retrieval-style schema and
    always decode raw media bytes before the caller-provided transform runs.
    This keeps downstream eval transforms working on PIL images instead of raw
    storage bytes.
    """

    def __init__(
        self,
        lance_path: str,
        read_columns: list[str] | None = None,
        extensions: list[LanceTableExtension] | None = None,
        metadata: dict[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        columns: list[str] | None = None,
        lazy: bool = True,
    ) -> None:
        if read_columns is None and columns is None:
            effective_read_columns = ["id", "text", "images", "media_metadata"]
        else:
            effective_read_columns = _normalize_read_columns(read_columns, columns=columns)

        if transform is None:
            effective_transform = _decode_eval_record
        else:
            effective_transform = partial(
                _apply_transform_chain,
                transforms=(_decode_eval_record, transform),
            )

        super().__init__(
            lance_path=lance_path,
            read_columns=effective_read_columns,
            extensions=extensions,
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )


class VideoEvalLanceDataset(EvalLanceDataset):
    """Compatibility wrapper that applies sampled-frame eval transform.

    Why it still exists:
        Older call sites still construct ``VideoEvalLanceDataset`` directly.
        The new architecture no longer needs a dedicated subclass because the
        same behavior can be expressed as ``EvalLanceDataset + transform``.

    Removal condition:
        Remove this wrapper after all call sites switch to
        ``EvalLanceDataset(..., transform=build_video_eval_transform(...))`` or
        another explicit eval transform composition.
    """

    def __init__(
        self,
        lance_path: str,
        num_frames: int = 8,
        read_columns: list[str] | None = None,
        extensions: list[LanceTableExtension] | None = None,
        metadata: dict[str, Any] | None = None,
        sample_transform: SampleTransform | None = None,
        *,
        columns: list[str] | None = None,
        lazy: bool = True,
    ) -> None:
        sampling_transform = build_video_eval_transform(num_frames=num_frames)
        if sample_transform is None:
            effective_transform = sampling_transform
        else:
            effective_transform = partial(
                _apply_transform_chain,
                transforms=(sampling_transform, sample_transform),
            )

        super().__init__(
            lance_path=lance_path,
            read_columns=read_columns,
            extensions=extensions,
            metadata=metadata,
            transform=effective_transform,
            columns=columns,
            lazy=lazy,
        )
        self.num_frames = num_frames


# Backward-compatible aliases kept only for direct raw/eval naming parity.
BaseLanceDataset = LanceDataset
BaseStructuredSampleLanceDataset = SampleLanceDataset
BaseEvalLanceDataset = EvalLanceDataset


__all__ = [
    "SampleTransform",
    "ResolvedTrainDatasetSpec",
    "ResolvedRetrievalEvalSpec",
    "LanceTableExtension",
    "LanceDataset",
    "SampleLanceDataset",
    "EvalLanceDataset",
    "VideoEvalLanceDataset",
    "LanceKeyedRowResolver",
    "build_video_eval_transform",
    "BaseLanceDataset",
    "BaseStructuredSampleLanceDataset",
    "BaseEvalLanceDataset",
]
