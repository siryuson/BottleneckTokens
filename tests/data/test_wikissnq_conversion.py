from __future__ import annotations

import json
from pathlib import Path

import lance
from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.conversion.wikissnq_train import (
    ensure_wikissnq_train_indices,
    write_wikissnq_train_root,
)
from vlm2emb.data.datasets.wikissnq_train import WikissnqTrainDataset  # noqa: F401


def _build_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


def test_wikissnq_conversion_filters_eval_overlap(tmp_path: Path) -> None:
    query_root = tmp_path / "wiki-ss-nq"
    query_root.mkdir()
    train_rows = [
        {
            "query_id": "train_0",
            "query": "who needs to know about the jet stream",
            "positive_passages": [
                {"docid": "1000", "title": "Jet stream", "text": "Jet stream body"},
                {"docid": "1001", "title": "Jet current", "text": "Jet current body"},
            ],
            "negative_passages": [],
            "answers": ["pilots"],
        },
        {
            "query_id": "train_1",
            "query": "what is the capital of france",
            "positive_passages": [
                {"docid": "1002", "title": "Paris", "text": "Paris body"},
            ],
            "negative_passages": [{"docid": "9", "title": "Lyon", "text": "Lyon body"}],
            "answers": ["paris"],
        },
    ]
    (query_root / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows),
        encoding="utf-8",
    )

    corpus_root = tmp_path / "wiki-ss-corpus-new"
    image_paths = []
    for stem, color in [
        ("1000", (255, 0, 0)),
        ("1001", (0, 255, 0)),
        ("1002", (0, 0, 255)),
    ]:
        image_path = tmp_path / f"{stem}.png"
        _build_image(color).save(image_path)
        image_paths.append(str(image_path))
    dataset = Dataset.from_dict(
        {
            "image": image_paths,
            "docid": ["0", "1", "2"],
        }
    )
    DatasetDict({"train": dataset.cast_column("image", DatasetImage(decode=False))}).save_to_disk(str(corpus_root))

    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    lance.write_dataset(
        [
            {
                "id": "q0",
                "text": "Find the document image that can answer the given query: who needs to know about the jet stream\n",
                "images": [],
                "media_metadata": None,
            }
        ],
        str(eval_root / "queries.lance"),
        mode="create",
    )

    output_root = tmp_path / "wikissnq_root"
    summary = write_wikissnq_train_root(
        query_root=query_root,
        corpus_root=corpus_root,
        eval_root=eval_root,
        output_root=output_root,
        overwrite=True,
    )

    assert summary["image_rows"] == 3
    assert summary["splits"]["official_train_raw"] == 3
    assert summary["splits"]["official_train"] == 1
    assert summary["splits"]["official_train_without_mmeb_v2_eval"] == 1
    assert summary["exclusion_rows"] == 2

    assert lance.dataset(str(output_root / "data" / "images.lance")).count_rows() == 3
    assert lance.dataset(str(output_root / "data" / "official_train.lance")).count_rows() == 1
    assert lance.dataset(str(output_root / "data" / "official_train_raw.lance")).count_rows() == 3

    dataset = AutoDataset.get("wikissnq_train").from_config(
        {
            "path": str(output_root),
            "dataset_name": "Wiki-SS-NQ",
            "split": "official_train",
            "transform": {
                "query": {
                    "instruction": "Find the document image that can answer the given query:",
                    "instruction_body_separator": "space",
                    "trailing_newline": "ensure_single",
                },
                "positive": {
                    "instruction": "Represent the given image",
                    "visual_token_placement": "own_line",
                    "trailing_newline": "ensure_single",
                },
                "negative": {"trailing_newline": "ensure_single", "empty": "empty_multimodal_input"},
            },
        }
    )
    sample = dataset[0]
    assert sample["query"]["text"] == "Find the document image that can answer the given query: what is the capital of france\n"
    assert sample["positive"]["text"] == "<|image_pad|>\nRepresent the given image\n"
    assert sample["negative"] == {"text": "", "media": []}
    assert len(sample["positive"]["media"]) == 1
    assert sample["metadata"]["positive_docid"] == "1002"


def test_ensure_wikissnq_train_indices(tmp_path: Path) -> None:
    lance.write_dataset(
        [{"path": "0", "image": b"x"}],
        str(tmp_path / "data" / "images.lance"),
        mode="create",
    )

    ensure_wikissnq_train_indices(tmp_path)

    image_indices = lance.dataset(str(tmp_path / "data" / "images.lance")).list_indices()
    assert any(index["fields"] == ["path"] for index in image_indices)
