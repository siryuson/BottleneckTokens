"""Country211 training conversion utilities."""

from __future__ import annotations

import shutil
import tarfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

SUPPORTED_SPLITS = ("official_train", "official_test", "official_train_without_mmeb_v2_eval")

SPLIT_SCHEMA = pa.schema(
    [
        pa.field("image_path", pa.string()),
        pa.field("source_tar", pa.string()),
        pa.field("sample_key", pa.string()),
        pa.field("class_index", pa.int32()),
        pa.field("class_name", pa.string()),
    ]
)
IMAGES_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("image", pa.binary()),
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


def read_country211_classnames(raw_root: Path) -> list[str]:
    """Read the Country211 class name list."""

    path = raw_root / "classnames.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Country211 classnames file: {path}")
    classnames = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classnames:
        raise ValueError(f"Country211 classnames file is empty: {path}")
    return classnames


def _split_tars(raw_root: Path, split_dir: str) -> list[Path]:
    """Return sorted shard tar paths for one split."""

    split_root = raw_root / split_dir
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing Country211 split directory: {split_root}")
    tar_paths = sorted(split_root.glob("*.tar"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    if not tar_paths:
        raise FileNotFoundError(f"Country211 split has no tar shards: {split_root}")
    return tar_paths


def _read_cls_value(tar: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    """Read one WebDataset class index member."""

    file_obj = tar.extractfile(member)
    if file_obj is None:
        raise ValueError(f"Unable to read Country211 class member: {member.name}")
    payload = file_obj.read().strip()
    if not payload:
        raise ValueError(f"Empty Country211 class member: {member.name}")
    return int(payload.decode("utf-8"))


def _split_rows_from_tar(
    *,
    raw_root: Path,
    split_dir: str,
    tar_path: Path,
    classnames: list[str],
) -> list[dict[str, Any]]:
    """Read split metadata rows from one Country211 tar shard."""

    rows: list[dict[str, Any]] = []
    source_tar = str(tar_path.relative_to(raw_root))
    with tarfile.open(tar_path) as tar:
        cls_by_key: dict[str, int] = {}
        has_image: set[str] = set()
        for member in tar:
            member_path = Path(member.name)
            sample_key = member_path.stem
            suffix = member_path.suffix.lower()
            if suffix == ".cls":
                cls_by_key[sample_key] = _read_cls_value(tar, member)
            elif suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                has_image.add(sample_key)
        for sample_key in sorted(has_image):
            class_index = cls_by_key.get(sample_key)
            if class_index is None:
                raise ValueError(f"Country211 sample lacks class label: {source_tar}:{sample_key}")
            if class_index < 0 or class_index >= len(classnames):
                raise ValueError(f"Country211 class index out of range: {source_tar}:{sample_key} -> {class_index}")
            image_path = f"{split_dir}/{tar_path.stem}/{sample_key}.jpg"
            rows.append(
                {
                    "image_path": image_path,
                    "source_tar": source_tar,
                    "sample_key": sample_key,
                    "class_index": class_index,
                    "class_name": classnames[class_index],
                }
            )
    return rows


def load_country211_split_rows(raw_root: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Return official train/test rows plus the eval-excluded train view."""

    classnames = read_country211_classnames(raw_root)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for tar_path in _split_tars(raw_root, "train"):
        train.extend(_split_rows_from_tar(raw_root=raw_root, split_dir="train", tar_path=tar_path, classnames=classnames))
    for tar_path in _split_tars(raw_root, "test"):
        test.extend(_split_rows_from_tar(raw_root=raw_root, split_dir="test", tar_path=tar_path, classnames=classnames))
    eval_keys = {row["image_path"] for row in test}
    train_without_eval = [row for row in train if row["image_path"] not in eval_keys]
    return OrderedDict(
        [
            ("official_train", train),
            ("official_test", test),
            ("official_train_without_mmeb_v2_eval", train_without_eval),
        ]
    )


def _image_rows_from_tar(*, raw_root: Path, split_dir: str, tar_path: Path) -> Iterator[dict[str, Any]]:
    """Yield image rows from one Country211 WebDataset tar shard."""

    with tarfile.open(tar_path) as tar:
        for member in tar:
            suffix = Path(member.name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                raise ValueError(f"Unable to read Country211 image member: {member.name}")
            sample_key = Path(member.name).stem
            yield {
                "path": f"{split_dir}/{tar_path.stem}/{sample_key}.jpg",
                "image": file_obj.read(),
            }


def _image_batches(*, raw_root: Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Build image table batches from all Country211 shards."""

    rows: list[dict[str, Any]] = []
    for split_dir in ("train", "test"):
        for tar_path in _split_tars(raw_root, split_dir):
            for row in _image_rows_from_tar(raw_root=raw_root, split_dir=split_dir, tar_path=tar_path):
                rows.append(row)
                if len(rows) >= batch_size:
                    yield pa.RecordBatch.from_pylist(rows, schema=IMAGES_SCHEMA)
                    rows = []
    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=IMAGES_SCHEMA)


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield split rows as Arrow record batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_country211_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Country211 runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing Country211 image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_country211_root(
    *,
    raw_root: Path,
    output_root: Path,
    overwrite: bool = False,
    image_batch_size: int = 1024,
    split_batch_size: int = 4096,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert Country211 into a raw-preserving Lance training root."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    split_rows = load_country211_split_rows(raw_root)
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(IMAGES_SCHEMA, _image_batches(raw_root=raw_root, batch_size=image_batch_size)),
        str(output_root / "data" / "images.lance"),
        schema=IMAGES_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    for split_name, rows in split_rows.items():
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(rows, batch_size=split_batch_size)),
            str(output_root / "data" / f"{split_name}.lance"),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            SPLIT_SCHEMA,
            _split_batches(split_rows["official_test"], batch_size=split_batch_size),
        ),
        str(output_root / "data" / "exclusions" / "mmeb_v2_eval.lance"),
        schema=SPLIT_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )

    ensure_country211_indices(output_root)
    return {
        "output_root": str(output_root),
        "image_rows": sum(len(rows) for name, rows in split_rows.items() if name in {"official_train", "official_test"}),
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "exclusion_rows": len(split_rows["official_test"]),
        "artifact_format": "country211_train_raw_lance_v1",
    }


__all__ = [
    "IMAGES_SCHEMA",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "ensure_country211_indices",
    "load_country211_split_rows",
    "read_country211_classnames",
    "write_country211_root",
]
