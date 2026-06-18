from __future__ import annotations

import io
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from vlm2emb.data.conversion.textvqa_train import write_textvqa_train_root
from vlm2emb.data.datasets.textvqa_train import TextvqaTrainDataset


def _make_jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_lance(path: Path, rows: list[dict[str, object]]) -> None:
    lance.write_dataset(pa.Table.from_pylist(rows), str(path), mode="create")


def _textvqa_row(
    *,
    image_id: str,
    question_id: int,
    question: str,
    answers: list[str],
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "question_id": question_id,
        "question": question,
        "question_tokens": question.rstrip("?").split(),
        "image": {"bytes": _make_jpeg_bytes((255, 0, 0)), "path": None},
        "image_width": 8,
        "image_height": 8,
        "flickr_original_url": "http://example.com/original",
        "flickr_300k_url": "http://example.com/300k",
        "answers": answers,
        "image_classes": ["phone"],
        "set_name": "train",
        "ocr_tokens": ["NOKIA"],
    }


def _write_textvqa_fixture(raw_root: Path, eval_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    train_rows = [
        _textvqa_row(
            image_id="img0",
            question_id=0,
            question="what is the brand of phone?",
            answers=["nokia", "nokia"],
        ),
        _textvqa_row(
            image_id="img1",
            question_id=1,
            question="what color is the sign?",
            answers=["red"],
        ),
    ]
    validation_rows = [
        _textvqa_row(image_id="img2", question_id=2, question="validation question?", answers=["yes"])
    ]
    test_rows = [_textvqa_row(image_id="img3", question_id=3, question="test question?", answers=["no"])]
    pq.write_table(pa.Table.from_pylist(train_rows), raw_root / "train-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(validation_rows), raw_root / "validation-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(test_rows), raw_root / "test-00000-of-00001.parquet")

    eval_root.mkdir(parents=True, exist_ok=True)
    _write_lance(
        eval_root / "queries.lance",
        [
            {
                "id": 0,
                "qry_text": "<|image_1|>\nRepresent the given image with the following question: what color is the sign?\n",
            }
        ],
    )
    _write_lance(eval_root / "candidates.lance", [{"id": 7, "text": "red"}])
    _write_lance(
        eval_root / "qrels.lance",
        [{"query_id": 0, "candidate_ids": [7], "candidate_scores": [1.0]}],
    )


def test_write_textvqa_root_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "textvqa"
    eval_root = tmp_path / "MMEB-V2" / "TextVQA"
    output_root = tmp_path / "TextVQA-out"
    _write_textvqa_fixture(raw_root, eval_root)

    summary = write_textvqa_train_root(raw_root=raw_root, output_root=output_root, eval_root=eval_root)

    assert summary["image_rows"] == 4
    assert summary["splits"]["official_train"] == 2
    assert summary["splits"]["official_train_without_mmeb_v2_eval"] == 1
    assert summary["exclusion_rows"] == 1
    assert not (output_root / "manifest.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    assert image_ds.schema.names == ["path", "image"]
    assert "image" not in train_ds.schema.names
    assert "image_id" in train_ds.schema.names
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = TextvqaTrainDataset(path=str(output_root), split="official_train_without_mmeb_v2_eval")
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|image_pad|>\nRepresent the given image with the following question: what is the brand of phone?\n"
    )
    assert sample["positive"]["text"] == "nokia"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["ocr_tokens"] == ["NOKIA"]
    assert sample["query"]["media"][0]["kind"] == "image"
