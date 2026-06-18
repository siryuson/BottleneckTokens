from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from PIL import Image

from vlm2emb.data.conversion.vizwiz_train import load_vizwiz_split_rows, write_vizwiz_root
from vlm2emb.data.datasets.vizwiz_train import VizwizTrainDataset


def _make_jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_lance(path: Path, rows: list[dict[str, object]]) -> None:
    lance.write_dataset(pa.Table.from_pylist(rows), str(path), mode="create")


def _answer(value: str, confidence: str = "yes") -> dict[str, str]:
    return {"answer": value, "answer_confidence": confidence}


def _write_vizwiz_fixture(raw_root: Path, eval_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    annotations = {
        "train.json": [
            {
                "image": "VizWiz_train_00000000.jpg",
                "question": "What is this product?",
                "answers": [_answer("basil leaves"), _answer("basil"), _answer("basil leaves", "maybe")],
                "answer_type": "other",
                "answerable": 1,
            },
            {
                "image": "VizWiz_train_00000001.jpg",
                "question": "What color is it?",
                "answers": [_answer("red")],
                "answer_type": "other",
                "answerable": 1,
            },
            {
                "image": "VizWiz_train_00000002.jpg",
                "question": "What does the label say?",
                "answers": [_answer("unanswerable")],
                "answer_type": "unanswerable",
                "answerable": 0,
            },
        ],
        "val.json": [
            {
                "image": "VizWiz_val_00000000.jpg",
                "question": "Can you read this?",
                "answers": [_answer("night time"), _answer("night time", "maybe")],
                "answer_type": "other",
                "answerable": 1,
            }
        ],
        "test.json": [{"image": "VizWiz_test_00000000.jpg", "question": "What is this?"}],
    }
    with zipfile.ZipFile(raw_root / "Annotations.zip", "w") as archive:
        for name, rows in annotations.items():
            archive.writestr(name, json.dumps(rows))

    with zipfile.ZipFile(raw_root / "train.zip", "w") as archive:
        for index in range(3):
            archive.writestr(f"train/VizWiz_train_0000000{index}.jpg", _make_jpeg_bytes((255, 0, 0)))
    with zipfile.ZipFile(raw_root / "val.zip", "w") as archive:
        archive.writestr("val/VizWiz_val_00000000.jpg", _make_jpeg_bytes((0, 255, 0)))

    eval_root.mkdir(parents=True, exist_ok=True)
    _write_lance(
        eval_root / "queries.lance",
        [
            {
                "id": 0,
                "qry_text": "<|image_1|>\nRepresent the given image with the following question: What color is it?\n",
            }
        ],
    )
    _write_lance(eval_root / "candidates.lance", [{"id": 7, "text": "red"}])
    _write_lance(
        eval_root / "qrels.lance",
        [{"query_id": 0, "candidate_ids": [7], "candidate_scores": [1.0]}],
    )


def test_load_vizwiz_split_rows_filters_answerable_and_eval_overlap(tmp_path: Path):
    raw_root = tmp_path / "vizwiz"
    eval_root = tmp_path / "MMEB-V2" / "VizWiz"
    _write_vizwiz_fixture(raw_root, eval_root)

    rows_by_split = load_vizwiz_split_rows(
        raw_root,
        eval_overlap_keys={("what color is it?", "red")},
    )

    assert rows_by_split["official_train"][0]["selected_answer"] == "basil leaves"
    assert len(rows_by_split["official_train"]) == 3
    assert len(rows_by_split["official_train_answerable"]) == 2
    assert len(rows_by_split["official_train_answerable_without_mmeb_v2_eval"]) == 1


def test_write_vizwiz_root_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "vizwiz"
    eval_root = tmp_path / "MMEB-V2" / "VizWiz"
    output_root = tmp_path / "VizWiz-out"
    _write_vizwiz_fixture(raw_root, eval_root)

    summary = write_vizwiz_root(raw_root=raw_root, output_root=output_root, eval_root=eval_root)

    assert summary["splits"]["official_train"] == 3
    assert summary["splits"]["official_train_answerable"] == 2
    assert summary["splits"]["official_train_answerable_without_mmeb_v2_eval"] == 1
    assert summary["exclusion_rows"] == 1
    assert summary["image_rows"] == 4
    assert not (output_root / "manifest.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_answerable_without_mmeb_v2_eval.lance"))
    assert image_ds.schema.names == ["path", "image"]
    assert "image" in train_ds.schema.names
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = VizwizTrainDataset(path=str(output_root), split="official_train_answerable_without_mmeb_v2_eval")
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|image_pad|>\nRepresent the given image with the following question: What is this product?\n"
    )
    assert sample["positive"]["text"] == "basil leaves"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["answer_type"] == "other"
    assert sample["query"]["media"][0]["kind"] == "image"
