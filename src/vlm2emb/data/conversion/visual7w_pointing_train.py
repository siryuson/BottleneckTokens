"""Visual7W Pointing training conversion utilities."""

from __future__ import annotations

import shutil
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
from PIL import Image

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

SUPPORTED_SPLITS = (
    "official_train",
    "official_val",
    "official_test",
    "official_train_without_mmeb_v2_eval",
)
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("qa_id", pa.int64()),
        pa.field("image_id", pa.int64()),
        pa.field("image_filename", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answer_box_id", pa.int64()),
        pa.field("answer_name", pa.string()),
        pa.field("question_type", pa.string()),
        pa.field("query_image_path", pa.string()),
        pa.field("positive_image_path", pa.string()),
        pa.field("crop_x", pa.int32()),
        pa.field("crop_y", pa.int32()),
        pa.field("crop_width", pa.int32()),
        pa.field("crop_height", pa.int32()),
    ]
)
IMAGE_SCHEMA = pa.schema([pa.field("path", pa.string()), pa.field("image", pa.binary())])


def _batched(items: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield fixed-size row batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_json_from_zip(zip_path: Path) -> dict[str, Any]:
    """Load the single JSON payload from one Visual7W zip file."""

    with zipfile.ZipFile(zip_path) as zf:
        json_name = next(name for name in zf.namelist() if name.endswith(".json"))
        return zf.read(json_name)


def _load_json_payload_from_zip(zip_path: Path) -> dict[str, Any]:
    """Load and parse JSON from one Visual7W zip file."""

    import json

    return json.loads(_load_json_from_zip(zip_path))


def load_visual7w_pointing_payload(raw_root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load pointing annotations and grounding boxes from raw zip files."""

    pointing_path = raw_root / "dataset_v7w_pointing.zip"
    grounding_path = raw_root / "dataset_v7w_grounding_annotations.zip"
    if not pointing_path.is_file():
        raise FileNotFoundError(f"Missing pointing annotations zip: {pointing_path}")
    if not grounding_path.is_file():
        raise FileNotFoundError(f"Missing grounding annotations zip: {grounding_path}")

    pointing_payload = _load_json_payload_from_zip(pointing_path)
    grounding_payload = _load_json_payload_from_zip(grounding_path)
    images = pointing_payload.get("images")
    boxes = grounding_payload.get("boxes")
    if not isinstance(images, list):
        raise TypeError("Visual7W pointing payload must contain an 'images' list.")
    if not isinstance(boxes, list):
        raise TypeError("Visual7W grounding payload must contain a 'boxes' list.")
    return images, {int(box["box_id"]): box for box in boxes}


def _clip_crop_box(box: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    """Clip one Visual7W box to source image bounds."""

    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    w = max(1, int(box["width"]))
    h = max(1, int(box["height"]))
    right = min(width, x + w)
    bottom = min(height, y + h)
    if right <= x or bottom <= y:
        raise ValueError(f"Invalid crop box after clipping: x={x}, y={y}, width={w}, height={h}")
    return x, y, right, bottom


def _encode_jpeg(image: Image.Image) -> bytes:
    """Encode one PIL image as JPEG bytes."""

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _image_member(image_filename: str) -> str:
    """Return the Visual7W image zip member for one filename."""

    return f"images/{image_filename}"


def build_visual7w_pointing_split_rows(raw_root: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Build raw-derived split rows for Visual7W Pointing."""

    images, box_by_id = load_visual7w_pointing_payload(raw_root)
    split_rows: OrderedDict[str, list[dict[str, Any]]] = OrderedDict(
        (name, []) for name in ("official_train", "official_val", "official_test")
    )
    for image_row in images:
        split_name = f"official_{image_row['split']}"
        if split_name not in split_rows:
            continue
        image_id = int(image_row["image_id"])
        image_filename = str(image_row["filename"])
        for qa in image_row["qa_pairs"]:
            answer_box_id = int(qa["answer"])
            answer_box = box_by_id.get(answer_box_id)
            if answer_box is None:
                raise KeyError(f"Missing grounding box '{answer_box_id}' for image '{image_filename}'.")
            crop_x = int(answer_box["x"])
            crop_y = int(answer_box["y"])
            crop_width = int(answer_box["width"])
            crop_height = int(answer_box["height"])
            split_rows[split_name].append(
                {
                    "sample_id": str(int(qa["qa_id"])),
                    "qa_id": int(qa["qa_id"]),
                    "image_id": image_id,
                    "image_filename": image_filename,
                    "question": str(qa["question"]),
                    "answer_box_id": answer_box_id,
                    "answer_name": str(answer_box.get("name", "") or ""),
                    "question_type": str(qa.get("type", "") or ""),
                    "query_image_path": f"image/{image_filename}",
                    "positive_image_path": f"crop/{answer_box_id}",
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                }
            )
    split_rows["official_train_without_mmeb_v2_eval"] = list(split_rows["official_train"])
    return split_rows


def _unique_sample_rows(split_rows: OrderedDict[str, list[dict[str, Any]]]) -> OrderedDict[str, dict[str, Any]]:
    """Collect unique rows by sample id across all splits."""

    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for split in ("official_train", "official_val", "official_test"):
        for row in split_rows[split]:
            rows.setdefault(str(row["sample_id"]), row)
    return rows


def _image_batches(
    *,
    raw_root: Path,
    split_rows: OrderedDict[str, list[dict[str, Any]]],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield full-image and answer-crop rows for the shared image table."""

    images_zip_path = raw_root / "visual7w_images.zip"
    if not images_zip_path.is_file():
        raise FileNotFoundError(f"Missing images zip: {images_zip_path}")
    _, box_by_id = load_visual7w_pointing_payload(raw_root)
    unique_rows = _unique_sample_rows(split_rows)
    rows_by_image: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in unique_rows.values():
        rows_by_image.setdefault(str(row["image_filename"]), []).append(row)

    emitted_rows: list[dict[str, Any]] = []
    seen_full_images: set[str] = set()
    seen_crops: set[int] = set()
    with zipfile.ZipFile(images_zip_path) as image_zip:
        for image_filename, sample_rows in rows_by_image.items():
            image_bytes = image_zip.read(_image_member(image_filename))
            if image_filename not in seen_full_images:
                emitted_rows.append({"path": f"image/{image_filename}", "image": image_bytes})
                seen_full_images.add(image_filename)
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
                for sample_row in sample_rows:
                    answer_box_id = int(sample_row["answer_box_id"])
                    if answer_box_id in seen_crops:
                        continue
                    left, top, right, bottom = _clip_crop_box(box_by_id[answer_box_id], image.width, image.height)
                    emitted_rows.append(
                        {
                            "path": f"crop/{answer_box_id}",
                            "image": _encode_jpeg(image.crop((left, top, right, bottom))),
                        }
                    )
                    seen_crops.add(answer_box_id)
            if len(emitted_rows) >= batch_size:
                yield pa.RecordBatch.from_pylist(emitted_rows, schema=IMAGE_SCHEMA)
                emitted_rows = []
    if emitted_rows:
        yield pa.RecordBatch.from_pylist(emitted_rows, schema=IMAGE_SCHEMA)


def _split_batches(rows: Iterable[dict[str, Any]], *, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Yield Visual7W Pointing split rows as Arrow batches."""

    for batch in _batched(rows, batch_size):
        yield pa.RecordBatch.from_pylist(batch, schema=SPLIT_SCHEMA)


def _create_scalar_index(dataset_path: Path, column: str) -> None:
    """Create one scalar index when it is absent."""

    dataset = lance.dataset(str(dataset_path))
    if any(column in index.field_names for index in dataset.describe_indices()):
        return
    dataset.create_scalar_index(column, "BTREE")


def ensure_visual7w_pointing_train_indices(output_root: Path) -> None:
    """Ensure lookup indices required by the Visual7W Pointing runtime."""

    images_path = output_root / "data" / "images.lance"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing Visual7W Pointing image table: {images_path}")
    _create_scalar_index(images_path, "path")


def write_visual7w_pointing_train_root(
    *,
    raw_root: Path,
    output_root: Path,
    overwrite: bool = False,
    split_batch_size: int = 4096,
    image_batch_size: int = 1024,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> dict[str, Any]:
    """Convert Visual7W Pointing into Lance training tables."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data" / "exclusions").mkdir(parents=True, exist_ok=True)

    split_rows = build_visual7w_pointing_split_rows(raw_root)
    split_counts: dict[str, int] = {}
    for split_name, rows in split_rows.items():
        path = output_root / "data" / f"{split_name}.lance"
        lance.write_dataset(
            pa.RecordBatchReader.from_batches(SPLIT_SCHEMA, _split_batches(rows, batch_size=split_batch_size)),
            str(path),
            schema=SPLIT_SCHEMA,
            mode="create",
            max_bytes_per_file=max_bytes_per_file,
        )
        split_counts[split_name] = lance.dataset(str(path)).count_rows()

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
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            IMAGE_SCHEMA,
            _image_batches(raw_root=raw_root, split_rows=split_rows, batch_size=image_batch_size),
        ),
        str(output_root / "data" / "images.lance"),
        schema=IMAGE_SCHEMA,
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
    )
    ensure_visual7w_pointing_train_indices(output_root)
    return {
        "output_root": str(output_root),
        "artifact_format": "visual7w_pointing_train_raw_lance_v1",
        "splits": split_counts,
        "exclusion_rows": split_counts["official_test"],
        "image_rows": lance.dataset(str(output_root / "data" / "images.lance")).count_rows(),
    }


__all__ = [
    "IMAGE_SCHEMA",
    "SPLIT_SCHEMA",
    "SUPPORTED_SPLITS",
    "build_visual7w_pointing_split_rows",
    "ensure_visual7w_pointing_train_indices",
    "load_visual7w_pointing_payload",
    "write_visual7w_pointing_train_root",
]
