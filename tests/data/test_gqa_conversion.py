from __future__ import annotations

import io
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from vlm2emb.data.conversion.gqa_train import write_gqa_train_root
from vlm2emb.data.datasets.gqa_train import GqaTrainDataset


def _make_jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_lance(path: Path, rows: list[dict[str, object]]) -> None:
    lance.write_dataset(pa.Table.from_pylist(rows), str(path), mode="create")


def _write_gqa_fixture(raw_root: Path, eval_root: Path) -> None:
    (raw_root / "train_all_instructions").mkdir(parents=True, exist_ok=True)
    (raw_root / "train_balanced_instructions").mkdir(parents=True, exist_ok=True)
    (raw_root / "train_all_images").mkdir(parents=True, exist_ok=True)
    instruction_rows = [
        {
            "id": "q0",
            "imageId": "img0",
            "question": "What is on the wall?",
            "answer": "pipe",
            "fullAnswer": "The pipe is on the wall.",
            "isBalanced": True,
            "groups": {"global": "indoor", "local": "wall"},
            "entailed": "[]",
            "equivalent": "[]",
            "types": {"structural": "query", "semantic": "rel", "detailed": "relS"},
            "annotations": {"question": [], "answer": [], "fullAnswer": []},
            "semantic": [],
            "semanticStr": "select",
        },
        {
            "id": "q1",
            "imageId": "missing",
            "question": "Missing image?",
            "answer": "nothing",
            "fullAnswer": "Nothing.",
            "isBalanced": False,
            "groups": {"global": "", "local": ""},
            "entailed": "[]",
            "equivalent": "[]",
            "types": {"structural": "query", "semantic": "attr", "detailed": "attr"},
            "annotations": {"question": [], "answer": [], "fullAnswer": []},
            "semantic": [],
            "semanticStr": "",
        },
    ]
    image_rows = [{"id": "img0", "image": {"bytes": _make_jpeg_bytes((255, 0, 0)), "path": None}}]
    pq.write_table(pa.Table.from_pylist(instruction_rows), raw_root / "train_all_instructions" / "train-000.parquet")
    pq.write_table(
        pa.Table.from_pylist([instruction_rows[0]]),
        raw_root / "train_balanced_instructions" / "train-000.parquet",
    )
    pq.write_table(pa.Table.from_pylist(image_rows), raw_root / "train_all_images" / "train-000.parquet")

    eval_root.mkdir(parents=True, exist_ok=True)
    _write_lance(
        eval_root / "queries.lance",
        [
            {
                "qry_img_path": "GQA/image_img0.jpg",
                "qry_text": "<|image_1|>\nRepresent the given image with the following question: What is on the wall?\n",
            }
        ],
    )


def test_write_gqa_root_filters_missing_and_eval_overlap(tmp_path: Path):
    raw_root = tmp_path / "GQA"
    eval_root = tmp_path / "MMEB-V2" / "GQA"
    output_root = tmp_path / "GQA-out"
    _write_gqa_fixture(raw_root, eval_root)

    summary = write_gqa_train_root(raw_root=raw_root, output_root=output_root, eval_root=eval_root)

    assert summary["image_rows"] == 1
    assert summary["splits"]["official_train_all"] == 1
    assert summary["splits"]["official_train_all_without_mmeb_v2_eval"] == 0
    assert summary["splits"]["official_train_balanced"] == 1
    assert summary["splits"]["official_train_balanced_without_mmeb_v2_eval"] == 0
    assert summary["exclusions"] == {"all": 1, "balanced": 1}
    assert not (output_root / "manifest.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_all.lance"))
    assert image_ds.schema.names == ["path", "image"]
    assert "question" in train_ds.schema.names
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = GqaTrainDataset(path=str(output_root), split="official_train_all")
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|image_pad|>\nRepresent the given image with the following question: What is on the wall?\n"
    )
    assert sample["positive"]["text"] == "The pipe is on the wall."
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["imageId"] == "img0"
    assert sample["query"]["media"][0]["kind"] == "image"
