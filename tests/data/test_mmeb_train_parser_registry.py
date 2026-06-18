from __future__ import annotations

import pytest

from tests.data.fixtures.mmeb_train_expected_content import (
    build_expected_content_document,
    build_expected_content_row,
)
from vlm2emb.data.datasets.mmeb_train import (
    MMEB_TRAIN_SUBSETS,
    PARSER_BY_SUBSET,
    get_mmeb_train_parser,
    parse_mmeb_train_content,
)


RAW_ROWS: dict[str, dict[str, object]] = {
    "ImageNet_1K": {
        "qry": "<|image_1|>\nRepresent the given image for classification\n",
        "qry_image_path": "images/ImageNet_1K/Train/image_0.jpg",
        "pos_text": "plane",
        "pos_image_path": "",
    },
    "OK-VQA": {
        "qry": "<|image_1|>\nRepresent the given image with the following question: What is shown?\n",
        "qry_image_path": "images/OK-VQA/Train/OK-VQA_image_0.jpg",
        "pos_text": "a dog",
        "pos_image_path": "",
        "neg_text": "a cat",
        "neg_image_path": "",
    },
    "MSCOCO_t2i": {
        "qry": "Find me an everyday image that matches the given caption: a small dog.\n",
        "qry_image_path": "",
        "pos_text": "<|image_1|>\nRepresent the given image.\n",
        "pos_image_path": "images/MSCOCO_t2i/Train/COCO.jpg",
    },
    "VisualNews_i2t": {
        "qry": "<|image_1|>\nFind a caption for the news in the given photo.\n",
        "qry_image_path": "images/VisualNews_i2t/Train/news.jpg",
        "pos_text": "A news caption.",
        "pos_image_path": "",
    },
    "CIRR": {
        "qry": "<|image_1|>\nGiven an image, find a similar everyday image with changes.\n",
        "qry_image_path": "images/CIRR/Train/query.jpg",
        "pos_text": "<|image_1|>\nRepresent the given image.\n",
        "pos_image_path": "images/CIRR/Train/positive.jpg",
    },
    "MSCOCO": {
        "qry": "<|image_1|>\nSelect the portion of the image that isolates the object.\n",
        "qry_image_path": "images/MSCOCO/Train/000000558840.jpg",
        "pos_text": "<|image_1|>\nRepresent the given cropped image of the object",
        "pos_image_path": "images/MSCOCO/Train/object.jpg",
    },
    "WebQA": {
        "qry": "<|image_1|>\nFind a Wikipedia image that answers this question: Which body part?\n",
        "qry_image_path": "",
        "pos_text": "<|image_1|>\nRepresent the given Wikipedia image.\n",
        "pos_image_path": "images/WebQA/Train/30272327.jpg",
    },
    "VisDial": {
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
}


def _expected_side(expected_content_side: dict[str, object]) -> dict[str, object]:
    return {
        "text": expected_content_side["text"],
        "path": expected_content_side["path"],
        "has_media": expected_content_side["has_media"],
    }


def test_mmeb_train_registry_covers_known_raw_train_subsets() -> None:
    assert tuple(PARSER_BY_SUBSET) == MMEB_TRAIN_SUBSETS
    assert set(PARSER_BY_SUBSET) == set(MMEB_TRAIN_SUBSETS)


def test_mmeb_train_parser_registry_matches_temporary_expected_content_table() -> None:
    expected_content = build_expected_content_document(
        [
            build_expected_content_row(raw_row, subset=subset)
            for subset, raw_row in RAW_ROWS.items()
        ]
    )

    for expected_row in expected_content["rows"]:
        subset = expected_row["subset"]
        actual = parse_mmeb_train_content(RAW_ROWS[subset], subset=subset)

        assert actual["subset"] == subset
        assert actual["query"] == _expected_side(expected_row["query"])
        assert actual["positive"] == _expected_side(expected_row["positive"])
        assert actual["negative"] == _expected_side(expected_row["negative"])

    mscoco = parse_mmeb_train_content(RAW_ROWS["MSCOCO"], subset="MSCOCO")
    assert mscoco["positive"]["text"] == (
        "<|image_pad|>\nRepresent the given cropped image of the object\n"
    )

    visdial = parse_mmeb_train_content(RAW_ROWS["VisDial"], subset="VisDial")
    assert visdial["query"]["text"] == (
        "Represent the given dialogue about an image, which is used for image retrieval:\n"
        "Q:is this a child or adult\nA:adult\n"
    )
    assert visdial["negative"]["text"] == "<|image_pad|>\n"


def test_mmeb_train_parser_registry_rejects_unknown_subset() -> None:
    with pytest.raises(KeyError, match="Unknown MMEB-train subset"):
        get_mmeb_train_parser("UnknownSubset")


def test_mmeb_train_parser_rejects_unknown_transform_key() -> None:
    with pytest.raises(ValueError, match="Unsupported MMEB-train transform.query key"):
        parse_mmeb_train_content(
            RAW_ROWS["WebQA"],
            subset="WebQA",
            transform={"query": {"unknown": "x"}},
        )


def test_mmeb_train_parser_rejects_unknown_transform_side() -> None:
    with pytest.raises(ValueError, match="Unsupported MMEB-train transform sides"):
        parse_mmeb_train_content(
            RAW_ROWS["WebQA"],
            subset="WebQA",
            transform={"candidate": {}},
        )


def test_mmeb_train_parser_rejects_non_mapping_transform() -> None:
    with pytest.raises(TypeError, match="transform kwargs must be a mapping"):
        parse_mmeb_train_content(
            RAW_ROWS["WebQA"],
            subset="WebQA",
            transform="bad",
        )


def test_mmeb_train_parser_rejects_non_mapping_side_transform() -> None:
    with pytest.raises(TypeError, match=r"transform\.query must be a mapping"):
        parse_mmeb_train_content(
            RAW_ROWS["WebQA"],
            subset="WebQA",
            transform={"query": "bad"},
        )


def test_mmeb_train_parser_rejects_unknown_transform_value() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported MMEB-train transform\.query\.trailing_newline value",
    ):
        parse_mmeb_train_content(
            RAW_ROWS["WebQA"],
            subset="WebQA",
            transform={"query": {"trailing_newline": "bad"}},
        )


def test_mmeb_train_parser_empty_transform_clears_side() -> None:
    raw_row = dict(RAW_ROWS["WebQA"])
    raw_row["neg_text"] = "should be cleared"
    raw_row["neg_image_path"] = "images/WebQA/Train/negative.jpg"

    parsed = parse_mmeb_train_content(raw_row, subset="WebQA")

    assert parsed["negative"] == {"text": "", "path": None, "has_media": False}
