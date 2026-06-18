from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from tests.data.fixtures.mmeb_train_expected_content import (
    build_expected_content_from_root,
    build_expected_content_row,
    write_expected_content_document,
)
from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN


def _write_parquet(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_expected_content_row_preserves_origin_content_and_repairs_webqa() -> None:
    row = build_expected_content_row(
        {
            "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: Which body part?\n",
            "qry_image_path": "",
            "pos_text": "<|image_1|>\nRepresent the given Wikipedia image.\n",
            "pos_image_path": "images/WebQA/Train/30272327.jpg",
        },
        subset="WebQA",
        source_path="/raw/WebQA/train-00000-of-00001.parquet",
        image_root="/raw",
    )

    assert row["query"] == {
        "text": "Find a Wikipedia image that answers this question: Which body part?\n",
        "path": None,
        "origin_path": None,
        "has_media": False,
    }
    assert row["positive"] == {
        "text": f"{STANDARD_IMAGE_TOKEN}\nRepresent the given Wikipedia image.\n",
        "path": "images/WebQA/Train/30272327.jpg",
        "origin_path": "/raw/images/WebQA/Train/30272327.jpg",
        "has_media": True,
    }
    assert row["negative"] == {
        "text": "",
        "path": None,
        "origin_path": None,
        "has_media": False,
    }


def test_expected_content_row_restores_visdial_empty_negative_token() -> None:
    row = build_expected_content_row(
        {
            "qry": (
                "Represent the given dialogue about an image, which is used for image retrieval: "
                "Q:is this a child or adult\nA:adult\n\n"
            ),
            "qry_image_path": "",
            "pos_text": "<|image_1|>\nRepresent the given image\n",
            "pos_image_path": "images/VisDial/Train/VisDial_image_0.jpg",
            "neg_text": "",
            "neg_image_path": "images/VisDial/Train/VisDial_image_61616.jpg",
        },
        subset="VisDial",
        image_root="/raw",
    )

    assert row["query"]["text"] == (
        "Represent the given dialogue about an image, which is used for image retrieval:\n"
        "Q:is this a child or adult\nA:adult\n"
    )
    assert row["negative"]["text"] == f"{STANDARD_IMAGE_TOKEN}\n"
    assert row["negative"]["path"] == "images/VisDial/Train/VisDial_image_61616.jpg"
    assert row["negative"]["origin_path"] == "/raw/images/VisDial/Train/VisDial_image_61616.jpg"
    assert row["negative"]["has_media"] is True


def test_expected_content_row_normalizes_visual_prompt_blocks() -> None:
    row = build_expected_content_row(
        {
            "qry": "<|image_1|>\nSelect the portion of the image.\n",
            "qry_image_path": "images/MSCOCO/Train/000000558840.jpg",
            "pos_text": "<|image_1|>\nRepresent the given cropped image of the object",
            "pos_image_path": "images/MSCOCO/Train/object.jpg",
        },
        subset="MSCOCO",
    )

    assert row["query"]["text"] == f"{STANDARD_IMAGE_TOKEN}\nSelect the portion of the image.\n"
    assert row["positive"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given cropped image of the object\n"
    )


def test_expected_content_row_trims_chartqa_answer_tail_spaces() -> None:
    row = build_expected_content_row(
        {
            "qry": "<|image_1|>\nRepresent the given image with the following question: Value?\n",
            "qry_image_path": "images/ChartQA/Train/ChartQA_image_0.jpg",
            "pos_text": "10.5 ",
            "pos_image_path": "",
            "neg_text": "7.2 ",
            "neg_image_path": "",
        },
        subset="ChartQA",
    )

    assert row["positive"]["text"] == "10.5"
    assert row["negative"]["text"] == "7.2"


def test_expected_content_document_reads_first_raw_parquet_row_per_subset(tmp_path) -> None:
    _write_parquet(
        tmp_path / "ImageNet_1K" / "train-00000-of-00001.parquet",
        [
            {
                "qry": "<|image_1|>\nRepresent the given image for classification\n",
                "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
                "pos_text": "plane",
                "pos_image_path": "",
            },
            {
                "qry": "second row",
                "qry_image_path": "",
                "pos_text": "unused",
                "pos_image_path": "",
            },
        ],
    )
    _write_parquet(
        tmp_path / "MSCOCO_t2i" / "train-00000-of-00001.parquet",
        [
            {
                "qry": "Find an everyday image that matches the given caption.\n",
                "qry_image_path": "",
                "pos_text": "<|image_1|>\nRepresent the given image.\n",
                "pos_image_path": "images/MSCOCO_t2i/Train/COCO.jpg",
            }
        ],
    )

    document = build_expected_content_from_root(
        tmp_path,
        subsets=["ImageNet_1K", "MSCOCO_t2i"],
    )

    assert [row["subset"] for row in document["rows"]] == ["ImageNet_1K", "MSCOCO_t2i"]
    assert document["rows"][0]["query"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image for classification\n"
    )
    assert document["rows"][0]["positive"]["text"] == "plane"
    assert document["rows"][1]["query"]["has_media"] is False
    assert document["rows"][1]["positive"]["has_media"] is True

    output = tmp_path / "expected_content.json"
    write_expected_content_document(document, output)
    loaded = json.loads(output.read_text())
    assert loaded["rows"][0]["source_row"] == 0
