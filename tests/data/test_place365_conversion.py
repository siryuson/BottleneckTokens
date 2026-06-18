from __future__ import annotations

from io import BytesIO
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from PIL import Image

from vlm2emb.data.conversion.place365_train import (
    IMAGES_SCHEMA,
    SPLIT_SCHEMA,
    SUPPORTED_SPLITS,
    ensure_place365_indices,
    load_place365_split_rows,
    write_place365_root,
)
from vlm2emb.data.datasets.place365_train import Place365TrainDataset


def _make_jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_place365_fixture(raw_root: Path) -> None:
    (raw_root / "train" / "airfield").mkdir(parents=True, exist_ok=True)
    (raw_root / "train" / "aquarium").mkdir(parents=True, exist_ok=True)
    (raw_root / "val" / "airfield").mkdir(parents=True, exist_ok=True)
    (raw_root / "val" / "aquarium").mkdir(parents=True, exist_ok=True)

    (raw_root / "train" / "airfield" / "00000001.jpg").write_bytes(_make_jpeg_bytes((255, 0, 0)))
    (raw_root / "train" / "aquarium" / "00000002.jpg").write_bytes(_make_jpeg_bytes((0, 255, 0)))
    (raw_root / "val" / "airfield" / "Places365_val_00000001.jpg").write_bytes(_make_jpeg_bytes((0, 0, 255)))
    (raw_root / "val" / "aquarium" / "Places365_val_00000002.jpg").write_bytes(_make_jpeg_bytes((255, 255, 0)))

    (raw_root / "train.txt").write_text(
        "train/airfield/00000001.jpg\ntrain/aquarium/00000002.jpg\n",
        encoding="utf-8",
    )
    (raw_root / "val.txt").write_text(
        "val/airfield/Places365_val_00000001.jpg\nval/aquarium/Places365_val_00000002.jpg\n",
        encoding="utf-8",
    )


def test_load_place365_split_rows_keeps_train_val_and_eval_excluded_view(tmp_path: Path):
    raw_root = tmp_path / "places365"
    _write_place365_fixture(raw_root)

    split_rows = load_place365_split_rows(raw_root)

    assert tuple(split_rows) == SUPPORTED_SPLITS
    assert len(split_rows["official_train"]) == 2
    assert len(split_rows["official_val"]) == 2
    assert len(split_rows["official_train_without_mmeb_v2_eval"]) == 2
    assert split_rows["official_train"][0]["class_name"] == "airfield"
    assert split_rows["official_val"][1]["image_path"].startswith("val/")


def test_write_place365_root_writes_raw_tables_and_loader_reads_train_split(tmp_path: Path):
    raw_root = tmp_path / "places365"
    output_root = tmp_path / "Place365"
    _write_place365_fixture(raw_root)

    payload = write_place365_root(raw_root=raw_root, output_root=output_root)

    assert payload["image_rows"] == 4
    assert payload["splits"]["official_train"] == 2
    assert payload["splits"]["official_val"] == 2
    assert payload["splits"]["official_train_without_mmeb_v2_eval"] == 2
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    assert image_ds.count_rows() == 4
    assert train_ds.count_rows() == 2
    assert image_ds.schema.names == ["path", "image"]
    assert train_ds.schema.names == ["image_path", "class_name"]
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = Place365TrainDataset(
        path=str(output_root),
        dataset_name="Place365",
        split="official_train_without_mmeb_v2_eval",
    )
    sample = dataset[0]

    assert sample["metadata"]["dataset_name"] == "Place365"
    assert sample["query"]["text"] == "<|image_pad|>\nIdentify the scene shown in the image\n"
    assert sample["positive"]["text"] == "airfield"
    assert sample["negative"]["text"] == ""
    assert len(sample["query"]["media"]) == 1
    assert sample["query"]["media"][0]["kind"] == "image"
    assert sample["metadata"]["class_name"] == "airfield"
    assert sample["metadata"]["split"] == "official_train_without_mmeb_v2_eval"


def test_ensure_place365_indices_adds_missing_persistent_index(tmp_path: Path):
    output_root = tmp_path / "Place365"
    (output_root / "data").mkdir(parents=True, exist_ok=True)

    image_rows = [
        {
            "path": "train/airfield/00000001.jpg",
            "image": _make_jpeg_bytes((255, 0, 0)),
        }
    ]
    split_rows = [
        {
            "image_path": "train/airfield/00000001.jpg",
            "class_name": "airfield",
        }
    ]

    lance.write_dataset(pa.Table.from_pylist(image_rows, schema=IMAGES_SCHEMA), str(output_root / "data" / "images.lance"), mode="create")
    lance.write_dataset(
        pa.Table.from_pylist(split_rows, schema=SPLIT_SCHEMA),
        str(output_root / "data" / "official_train.lance"),
        mode="create",
    )

    assert lance.dataset(str(output_root / "data" / "images.lance")).describe_indices() == []

    ensure_place365_indices(output_root)

    assert any("path" in idx.field_names for idx in lance.dataset(str(output_root / "data" / "images.lance")).describe_indices())
