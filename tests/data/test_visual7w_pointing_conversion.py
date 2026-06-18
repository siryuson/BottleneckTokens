from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from PIL import Image

from vlm2emb.data.conversion.visual7w_pointing_train import (
    IMAGE_SCHEMA,
    SPLIT_SCHEMA,
    SUPPORTED_SPLITS,
    build_visual7w_pointing_split_rows,
    ensure_visual7w_pointing_train_indices,
    write_visual7w_pointing_train_root,
)
from vlm2emb.data.datasets.visual7w_pointing import Visual7wPointingTrainDataset


def _make_jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((16, 16, 3), color, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)


def _write_visual7w_pointing_fixture(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_zip(
        raw_root / "visual7w_images.zip",
        {
            "images/v7w_1.jpg": _make_jpeg_bytes((255, 0, 0)),
            "images/v7w_2.jpg": _make_jpeg_bytes((0, 255, 0)),
            "images/v7w_3.jpg": _make_jpeg_bytes((0, 0, 255)),
        },
    )

    pointing_payload = {
        "dataset": "visual7w",
        "version": "1.0",
        "images": [
            {
                "image_id": 1,
                "split": "train",
                "filename": "v7w_1.jpg",
                "qa_pairs": [
                    {
                        "qa_id": 101,
                        "question": "Which box frames the toilet?",
                        "answer": 1001,
                        "multiple_choices": [1002, 1003, 1004],
                        "type": "which",
                    }
                ],
            },
            {
                "image_id": 2,
                "split": "val",
                "filename": "v7w_2.jpg",
                "qa_pairs": [
                    {
                        "qa_id": 201,
                        "question": "Which object is in the station?",
                        "answer": 2001,
                        "multiple_choices": [2002, 2003, 2004],
                        "type": "which",
                    }
                ],
            },
            {
                "image_id": 3,
                "split": "test",
                "filename": "v7w_3.jpg",
                "qa_pairs": [
                    {
                        "qa_id": 301,
                        "question": "Which pizza is on the left?",
                        "answer": 3001,
                        "multiple_choices": [3002, 3003, 3004],
                        "type": "which",
                    }
                ],
            },
        ],
    }
    _write_zip(
        raw_root / "dataset_v7w_pointing.zip",
        {"dataset_v7w_pointing.json": json.dumps(pointing_payload).encode("utf-8")},
    )

    grounding_payload = {
        "dataset": "visual7w",
        "version": "1.0",
        "boxes": [
            {"box_id": 1001, "name": "toilet", "height": 6, "width": 6, "y": 2, "x": 3},
            {"box_id": 1002, "name": "sink", "height": 6, "width": 6, "y": 0, "x": 0},
            {"box_id": 1003, "name": "wall", "height": 4, "width": 4, "y": 8, "x": 8},
            {"box_id": 1004, "name": "tile", "height": 5, "width": 5, "y": 10, "x": 10},
            {"box_id": 2001, "name": "vehicle", "height": 8, "width": 8, "y": 1, "x": 1},
            {"box_id": 2002, "name": "platform", "height": 5, "width": 5, "y": 2, "x": 8},
            {"box_id": 2003, "name": "light", "height": 4, "width": 4, "y": 6, "x": 6},
            {"box_id": 2004, "name": "sign", "height": 4, "width": 4, "y": 10, "x": 10},
            {"box_id": 3001, "name": "pizza", "height": 7, "width": 7, "y": 3, "x": 2},
            {"box_id": 3002, "name": "plate", "height": 6, "width": 6, "y": 0, "x": 0},
            {"box_id": 3003, "name": "table", "height": 5, "width": 5, "y": 9, "x": 9},
            {"box_id": 3004, "name": "slice", "height": 4, "width": 4, "y": 11, "x": 11},
        ],
    }
    _write_zip(
        raw_root / "dataset_v7w_grounding_annotations.zip",
        {"dataset_v7w_grounding_annotations.json": json.dumps(grounding_payload).encode("utf-8")},
    )


def test_build_visual7w_pointing_split_rows_keeps_official_splits(tmp_path: Path):
    raw_root = tmp_path / "visual7w"
    _write_visual7w_pointing_fixture(raw_root)

    split_rows = build_visual7w_pointing_split_rows(raw_root)

    assert tuple(split_rows) == SUPPORTED_SPLITS
    assert len(split_rows["official_train"]) == 1
    assert len(split_rows["official_val"]) == 1
    assert len(split_rows["official_test"]) == 1
    assert split_rows["official_train"][0]["question"] == "Which box frames the toilet?"
    assert split_rows["official_train"][0]["answer_name"] == "toilet"


def test_write_visual7w_pointing_root_writes_shared_tables_and_loader_reads_train_split(tmp_path: Path):
    raw_root = tmp_path / "visual7w"
    output_root = tmp_path / "Visual7W-Pointing"
    _write_visual7w_pointing_fixture(raw_root)

    payload = write_visual7w_pointing_train_root(raw_root=raw_root, output_root=output_root)

    assert payload["image_rows"] == 6
    assert payload["splits"]["official_train"] == 1
    assert payload["splits"]["official_val"] == 1
    assert payload["splits"]["official_test"] == 1
    assert payload["splits"]["official_train_without_mmeb_v2_eval"] == 1
    assert not (output_root / "README.md").exists()
    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "dataset_infos.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    assert image_ds.count_rows() == 6
    assert train_ds.count_rows() == 1
    assert image_ds.schema.names == ["path", "image"]
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = Visual7wPointingTrainDataset(
        path=str(output_root),
        split="official_train_without_mmeb_v2_eval",
    )
    sample = dataset[0]

    assert sample["metadata"]["dataset_name"] == "Visual7W-Pointing"
    assert sample["query"]["text"] == (
        "<|image_pad|>\nSelect the portion of the image that answers the question \"Which box frames the toilet?\"\n"
    )
    assert sample["positive"]["text"] == "<|image_pad|>\nRepresent the given cropped image of the object.\n"
    assert sample["negative"]["text"] == ""
    assert len(sample["query"]["media"]) == 1
    assert len(sample["positive"]["media"]) == 1
    assert sample["metadata"]["qa_id"] == 101
    assert sample["metadata"]["answer_name"] == "toilet"
    assert sample["metadata"]["split"] == "official_train_without_mmeb_v2_eval"


def test_ensure_visual7w_pointing_indices_adds_missing_persistent_indices(tmp_path: Path):
    output_root = tmp_path / "Visual7W-Pointing"
    (output_root / "data" / "splits").mkdir(parents=True, exist_ok=True)

    image_rows = [
        {"path": "image/v7w_1.jpg", "image": _make_jpeg_bytes((255, 0, 0))},
        {"path": "crop/1001", "image": _make_jpeg_bytes((255, 0, 0))},
    ]
    split_rows = [
        {
            "sample_id": "sample_1",
            "qa_id": 101,
            "image_id": 1,
            "image_filename": "v7w_1.jpg",
            "question": "Which box frames the toilet?",
            "answer_box_id": 1001,
            "answer_name": "toilet",
            "question_type": "which",
            "query_image_path": "image/v7w_1.jpg",
            "positive_image_path": "crop/1001",
            "crop_x": 3,
            "crop_y": 2,
            "crop_width": 6,
            "crop_height": 6,
        }
    ]

    lance.write_dataset(pa.Table.from_pylist(image_rows, schema=IMAGE_SCHEMA), str(output_root / "data" / "images.lance"), mode="create")
    lance.write_dataset(
        pa.Table.from_pylist(split_rows, schema=SPLIT_SCHEMA),
        str(output_root / "data" / "official_train.lance"),
        mode="create",
    )

    assert lance.dataset(str(output_root / "data" / "images.lance")).describe_indices() == []

    ensure_visual7w_pointing_train_indices(output_root)

    assert any("path" in idx.field_names for idx in lance.dataset(str(output_root / "data" / "images.lance")).describe_indices())
