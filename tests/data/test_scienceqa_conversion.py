from __future__ import annotations

import io
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from vlm2emb.data.conversion.scienceqa_train import write_scienceqa_train_root
from vlm2emb.data.datasets.scienceqa_train import ScienceqaTrainDataset


def _make_png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_split(raw_root: Path, split: str, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, raw_root / f"{split}-00000-of-00001.parquet")


def _scienceqa_row(
    *,
    question: str,
    answer: int,
    image: bytes | None,
) -> dict[str, object]:
    return {
        "image": None if image is None else {"bytes": image, "path": None},
        "question": question,
        "choices": ["red", "blue", "green"],
        "answer": answer,
        "hint": "hint",
        "task": "closed choice",
        "grade": "grade2",
        "subject": "natural science",
        "topic": "colors",
        "category": "test",
        "skill": "identify",
        "lecture": "lecture",
        "solution": "solution",
    }


def _write_scienceqa_fixture(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_split(
        raw_root,
        "train",
        [
            _scienceqa_row(question="What color is shown?", answer=1, image=_make_png_bytes((0, 0, 255))),
            _scienceqa_row(question="No image question?", answer=0, image=None),
        ],
    )
    _write_split(
        raw_root,
        "validation",
        [_scienceqa_row(question="Validation color?", answer=2, image=_make_png_bytes((0, 255, 0)))],
    )
    _write_split(
        raw_root,
        "test",
        [_scienceqa_row(question="Test color?", answer=0, image=_make_png_bytes((255, 0, 0)))],
    )


def test_write_scienceqa_root_and_runtime_sample(tmp_path: Path):
    raw_root = tmp_path / "ScienceQA-IMG"
    output_root = tmp_path / "ScienceQA"
    _write_scienceqa_fixture(raw_root)

    summary = write_scienceqa_train_root(raw_root=raw_root, output_root=output_root)

    assert summary["image_rows"] == 3
    assert summary["splits"]["official_train"] == 2
    assert summary["splits"]["official_test"] == 1
    assert not (output_root / "manifest.json").exists()

    image_ds = lance.dataset(str(output_root / "data" / "images.lance"))
    train_ds = lance.dataset(str(output_root / "data" / "official_train_without_mmeb_v2_eval.lance"))
    assert image_ds.schema.names == ["path", "image"]
    assert train_ds.schema.names == [
        "image_path",
        "question",
        "choices",
        "answer",
        "hint",
        "task",
        "grade",
        "subject",
        "topic",
        "category",
        "skill",
        "lecture",
        "solution",
    ]
    assert any("path" in idx.field_names for idx in image_ds.describe_indices())

    dataset = ScienceqaTrainDataset(path=str(output_root), split="official_train_without_mmeb_v2_eval")
    sample = dataset[0]
    no_image_sample = dataset[1]

    assert sample["query"]["text"] == (
        "<|image_pad|>\nRepresent the given image with the following question: What color is shown?\n"
    )
    assert sample["positive"]["text"] == "blue"
    assert sample["negative"]["text"] == ""
    assert sample["metadata"]["choices"] == ["red", "blue", "green"]
    assert sample["query"]["media"][0]["kind"] == "image"
    assert no_image_sample["query"]["text"] == (
        "Represent the given image with the following question: No image question?\n"
    )
    assert no_image_sample["query"]["media"] == []
