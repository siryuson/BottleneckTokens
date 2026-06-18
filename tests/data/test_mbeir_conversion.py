from __future__ import annotations

import json
from pathlib import Path

import lance
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.conversion.mbeir_train import ensure_mbeir_train_indices, write_mbeir_train_root
from vlm2emb.data.datasets.mbeir_train import MbeirTrainDataset  # noqa: F401


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path, format="JPEG")


def _build_raw_mbeir_fixture(root: Path) -> None:
    _write_jsonl(
        root / "query/train/mbeir_fashioniq_train.jsonl",
        [
            {
                "qid": "7:1",
                "query_txt": "is darker with red details.",
                "query_img_path": "mbeir_images/fashioniq_images/query.jpg",
                "query_modality": "image,text",
                "query_src_content": '{"candidate_img_id":"query"}',
                "pos_cand_list": ["7:10"],
                "neg_cand_list": [],
                "task_id": 7,
            },
            {
                "qid": "7:2",
                "query_txt": "excluded eval row.",
                "query_img_path": "mbeir_images/fashioniq_images/excluded.jpg",
                "query_modality": "image,text",
                "query_src_content": None,
                "pos_cand_list": ["7:11"],
                "neg_cand_list": [],
                "task_id": 7,
            },
        ],
    )
    _write_jsonl(
        root / "query/test/mbeir_fashioniq_task7_test.jsonl",
        [
            {
                "qid": "7:2",
                "query_txt": "excluded eval row.",
                "query_img_path": "mbeir_images/fashioniq_images/excluded.jpg",
                "query_modality": "image,text",
                "pos_cand_list": ["7:11"],
                "neg_cand_list": [],
                "task_id": 7,
            }
        ],
    )
    _write_jsonl(
        root / "cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl",
        [
            {
                "did": "7:10",
                "txt": None,
                "img_path": "mbeir_images/fashioniq_images/positive.jpg",
                "modality": "image",
                "src_content": '{"img_id":"positive"}',
            },
            {
                "did": "7:11",
                "txt": None,
                "img_path": "mbeir_images/fashioniq_images/excluded_positive.jpg",
                "modality": "image",
                "src_content": '{"img_id":"excluded_positive"}',
            },
        ],
    )
    _write_image(root / "mbeir_images/fashioniq_images/query.jpg", (250, 0, 0))
    _write_image(root / "mbeir_images/fashioniq_images/excluded.jpg", (0, 250, 0))
    _write_image(root / "mbeir_images/fashioniq_images/positive.jpg", (0, 0, 250))
    _write_image(root / "mbeir_images/fashioniq_images/excluded_positive.jpg", (250, 250, 0))


def test_mbeir_conversion_and_runtime(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "out"
    _build_raw_mbeir_fixture(raw_root)

    summary = write_mbeir_train_root(
        raw_root=raw_root,
        output_root=output_root,
        dataset_names=["FashionIQ_train"],
        overwrite=True,
        num_workers=2,
    )

    assert summary["datasets"]["FashionIQ_train"]["official_train"] == 2
    assert summary["datasets"]["FashionIQ_train"]["official_train_without_mmeb_v2_eval"] == 1
    assert summary["candidate_rows"]["FashionIQ_train"] == 2
    assert summary["image_rows"] == 4

    train = lance.dataset(str(output_root / "data/FashionIQ_train/official_train_without_mmeb_v2_eval.lance"))
    assert train.count_rows() == 1
    assert train.schema.names == [
        "qid",
        "query_txt",
        "query_img_path",
        "query_modality",
        "query_src_content",
        "pos_cand_list",
        "neg_cand_list",
        "task_id",
    ]

    ensure_mbeir_train_indices(output_root, ["FashionIQ_train"])

    dataset = AutoDataset.get("mbeir_train").from_config(
        {
            "path": str(output_root),
            "dataset_name": "FashionIQ_train",
            "split": "official_train_without_mmeb_v2_eval",
            "transform": {
                "query": {
                    "instruction": "Find an image to match the fashion image and style note:",
                    "instruction_body_separator": "space",
                    "visual_token_placement": "own_line",
                    "trailing_newline": "ensure_single",
                },
                "positive": {
                    "instruction": "Represent the given image.",
                    "instruction_body_separator": "space",
                    "visual_token_placement": "own_line",
                    "trailing_newline": "ensure_single",
                },
                "negative": {"trailing_newline": "ensure_single", "empty": "empty_multimodal_input"},
            },
        }
    )
    sample = dataset[0]

    assert sample["query"]["text"] == (
        "<|image_pad|>\n"
        "Find an image to match the fashion image and style note: "
        "is darker with red details.\n"
    )
    assert len(sample["query"]["media"]) == 1
    assert sample["positive"]["text"] == "<|image_pad|>\nRepresent the given image.\n"
    assert len(sample["positive"]["media"]) == 1
    assert sample["negative"] == {"text": "", "media": []}
    assert sample["metadata"]["positive_did"] == "7:10"
