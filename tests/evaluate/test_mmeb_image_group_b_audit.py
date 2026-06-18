from __future__ import annotations

import os

import io
import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from PIL import Image

from vlm2emb.data.datasets.const import STANDARD_IMAGE_TOKEN
from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import get_path_rel
from vlm2emb.data.datasets.mmeb_v2.dispatch import get_parser_module

MMEB_V2_REAL_ROOT = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2"))


GROUP_B_SUBSETS = [
    "MSCOCO_i2t",
    "VisualNews_i2t",
    "VisualNews_t2i",
    "Wiki-SS-NQ",
    "EDIS",
    "WebQA",
    "MSCOCO_t2i",
    "VisDial",
    "NIGHTS",
    "OVEN",
    "FashionIQ",
    "CIRR",
    "Visual7W-Pointing",
    "MSCOCO",
    "RefCOCO",
    "RefCOCO-Matching",
]

GROUP_B_REAL_EXPECTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "MSCOCO_i2t",
        "<|image_pad|>\nFind an image caption describing the given everyday image.\n",
        "A man with a red helmet on a small moped on a dirt road.",
        "<|image_pad|>\nFind an image caption describing the given everyday image.\n",
        "A man with a red helmet on a small moped on a dirt road.",
    ),
    (
        "VisualNews_i2t",
        "<|image_pad|>\nFind a caption for the news in the given photo.\n",
        "Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.",
        "<|image_pad|>\nFind a caption for the news in the given photo.\n",
        "Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.",
    ),
    (
        "VisualNews_t2i",
        "Retrieve an image of this news caption. Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.\n",
        "<|image_pad|>\nRepresent the given image.\n",
        "Retrieve an image of this news caption. Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.\n",
        "<|image_pad|>\nRepresent the given image.\n",
    ),
    (
        "Wiki-SS-NQ",
        "Find the document image that can answer the given query: who needs to know about the jet stream\n",
        "<|image_pad|>\nRepresent the given image\n",
        "Find the document image that can answer the given query: who needs to know about the jet stream\n",
        "<|image_pad|>\nRepresent the given image\n",
    ),
    (
        "EDIS",
        "Find a news image that matches the provided caption: Former Israeli president and prime minister Shimon Peres right and Palestinian President Mahmoud Abbas arrive at the World Economic Forum in Southern Shuneh Jordan in May 2015.\n",
        "<|image_pad|>\nRepresent the given image with related text information: WorldAs the West pays tribute to Peres, many Arabs recall a legacy of destruction.\n",
        "Find a news image that matches the provided caption: Former Israeli president and prime minister Shimon Peres right and Palestinian President Mahmoud Abbas arrive at the World Economic Forum in Southern Shuneh Jordan in May 2015.\n",
        "<|image_pad|>\nRepresent the given image with related text information: WorldAs the West pays tribute to Peres, many Arabs recall a legacy of destruction.\n",
    ),
    (
        "WebQA",
        "Find a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals in a cup shape?\n",
        "<|image_pad|>\nRepresent the given Wikipedia image with related text information: 2020-05-08 15 17 05 Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia.\n",
        "Find a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals in a cup shape?\n",
        "<|image_pad|>\nRepresent the given Wikipedia image with related text information: 2020-05-08 15 17 05 Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia.\n",
    ),
    (
        "MSCOCO_t2i",
        "Find me an everyday image that matches the given caption: A man with a red helmet on a small moped on a dirt road.\n",
        "<|image_pad|>\nRepresent the given image.\n",
        "Find me an everyday image that matches the given caption: A man with a red helmet on a small moped on a dirt road.\n",
        "<|image_pad|>\nRepresent the given image.\n",
    ),
    (
        "VisDial",
        "Represent the given dialogue about an image, which is used for image retrieval: Q:is the photo in color\nA:yes\nQ:is it a professional photo\nA:no\nQ:is it well lit\nA:no\nQ:is it daytime\nA:i don't see windows\nQ:does this look like an adults bedroom\nA:maybe\nQ:is the room clean\nA:no\nQ:can you tell what kind of computer it is\nA:no not really\nQ:is it a flat screen\nA:yes\nQ:what's the desk made out of\nA:cheap plastic or wood\nQ:is there a computer chair\nA:yes\n\n",
        "<|image_pad|>\nRepresent the given image\n",
        "Represent the given dialogue about an image, which is used for image retrieval:\nQ:is the photo in color\nA:yes\nQ:is it a professional photo\nA:no\nQ:is it well lit\nA:no\nQ:is it daytime\nA:i don't see windows\nQ:does this look like an adults bedroom\nA:maybe\nQ:is the room clean\nA:no\nQ:can you tell what kind of computer it is\nA:no not really\nQ:is it a flat screen\nA:yes\nQ:what's the desk made out of\nA:cheap plastic or wood\nQ:is there a computer chair\nA:yes\n",
        "<|image_pad|>\nRepresent the given image\n",
    ),
    (
        "NIGHTS",
        "<|image_pad|>\nFind a day-to-day image that looks similar to the provided image.\n",
        "<|image_pad|>\nRepresent the given image.\n",
        "<|image_pad|>\nFind a day-to-day image that looks similar to the provided image.\n",
        "<|image_pad|>\nRepresent the given image.\n",
    ),
    (
        "OVEN",
        "<|image_pad|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image: What is the name of this place?\n",
        "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Titisee. The Titisee is a lake in the southern Black Forest in Baden-Württemberg. It covers an area of 1.3 (km2) and is an average of 20 (m) deep. It owes its formation to the Feldberg glacier, the moraines of which were formed in the Pleistocene epoch and nowadays form the shores of the lake. The lake's outflow, at 840 (m) above sea level, is the River Gutach, which merges with the Haslach stream below Kappel to form the Wutach. The waters of the Titisee thus drain eventually into the Upper Rhine between Tiengen and Waldshut. On the north shore lies the.\n",
        "<|image_pad|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image: What is the name of this place?\n",
        "<|image_pad|>\nRepresent the given Wikipedia image with related text information: Titisee. The Titisee is a lake in the southern Black Forest in Baden-Württemberg. It covers an area of 1.3 (km2) and is an average of 20 (m) deep. It owes its formation to the Feldberg glacier, the moraines of which were formed in the Pleistocene epoch and nowadays form the shores of the lake. The lake's outflow, at 840 (m) above sea level, is the River Gutach, which merges with the Haslach stream below Kappel to form the Wutach. The waters of the Titisee thus drain eventually into the Upper Rhine between Tiengen and Waldshut. On the north shore lies the.\n",
    ),
    (
        "FashionIQ",
        "<|image_pad|>\nFind an image to match the fashion image and style note: Is shiny and silver with shorter sleeves and fit and flare.\n",
        "<|image_pad|>\nRepresent the given image.\n",
        "<|image_pad|>\nFind an image to match the fashion image and style note: Is shiny and silver with shorter sleeves and fit and flare.\n",
        "<|image_pad|>\nRepresent the given image.\n",
    ),
    (
        "CIRR",
        "<|image_pad|>\nGiven an image, find a similar everyday image with the described changes: Show three bottles of soft drink.\n",
        "<|image_pad|>\nRepresent the given image.\n",
        "<|image_pad|>\nGiven an image, find a similar everyday image with the described changes: Show three bottles of soft drink.\n",
        "<|image_pad|>\nRepresent the given image.\n",
    ),
    (
        "Visual7W-Pointing",
        "<|image_pad|>\nSelect the portion of the image that answers the question \"Which box frames the toilet?\"\n",
        "<|image_pad|>\nRepresent the given cropped image of the object",
        "<|image_pad|>\nSelect the portion of the image that answers the question \"Which box frames the toilet?\"\n",
        "<|image_pad|>\nRepresent the given cropped image of the object.\n",
    ),
    (
        "MSCOCO",
        "<|image_pad|>\nCrop the image to to isolate the object labeled as \"dog\"\n",
        "<|image_pad|>\nRepresent the given cropped image of the object",
        "<|image_pad|>\nCrop the image to to isolate the object labeled as \"dog\"\n",
        "<|image_pad|>\nRepresent the given cropped image of the object.\n",
    ),
    (
        "RefCOCO",
        "<|image_pad|>\nSelect the portion of the image that follows the language expressions. \"bowl behind the others can only see part\"\n",
        "<|image_pad|>\nRepresent the given cropped image of the object",
        "<|image_pad|>\nSelect the portion of the image that follows the language expressions. \"bowl behind the others can only see part\"\n",
        "<|image_pad|>\nRepresent the given cropped image of the object.\n",
    ),
    (
        "RefCOCO-Matching",
        "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"bowl behind the others can only see part\"\n",
        "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"Dish in top right corner\"\n",
        "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"bowl behind the others can only see part\"\n",
        "<|image_pad|>\nSelect the portion of the image that follows the language expressions: \"Dish in top right corner\"\n",
    ),
]

def _make_jpeg_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_eval_root(
    root,
    *,
    queries: list[dict[str, object]],
    candidates: list[dict[str, object]],
    qrels: list[dict[str, object]],
    metadata_name: str,
    task_type: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(pa.Table.from_pylist(queries), str(root / "queries.lance"), mode="create")
    lance.write_dataset(pa.Table.from_pylist(candidates), str(root / "candidates.lance"), mode="create")
    lance.write_dataset(pa.Table.from_pylist(qrels), str(root / "qrels.lance"), mode="create")
    with (root / "metadata.json").open("w") as file_handle:
        json.dump({"name": metadata_name, "task_type": task_type}, file_handle)


@pytest.mark.parametrize(
    ("subset_name", "expected_query_columns", "expected_candidate_columns"),
    [
        ("MSCOCO_i2t", ("id", "qry_text", "image"), ("id", "text")),
        ("VisualNews_i2t", ("id", "qry_text", "image"), ("id", "text")),
        ("VisualNews_t2i", ("id", "qry_text"), ("id", "tgt_text", "image")),
        ("Wiki-SS-NQ", ("id", "qry_text"), ("id", "tgt_text", "image")),
        ("EDIS", ("id", "qry_text"), ("id", "tgt_text", "image")),
        ("WebQA", ("id", "qry_text"), ("id", "tgt_inst", "tgt_text", "image")),
        ("MSCOCO_t2i", ("id", "qry_text"), ("id", "tgt_text", "image")),
        ("VisDial", ("id", "qry_inst", "qry_text"), ("id", "tgt_inst", "tgt_text", "tgt_img_path", "image")),
        ("NIGHTS", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("OVEN", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("FashionIQ", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("CIRR", ("id", "qry_inst", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("Visual7W-Pointing", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("MSCOCO", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("RefCOCO", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
        ("RefCOCO-Matching", ("id", "qry_text", "image"), ("id", "tgt_text", "image")),
    ],
)
def test_group_b_subset_contract_snapshot(subset_name: str, expected_query_columns: tuple[str, ...], expected_candidate_columns: tuple[str, ...]) -> None:
    parser_module = get_parser_module(subset_name)
    assert tuple(parser_module.get_query_read_columns(subset_name)) == expected_query_columns
    assert tuple(parser_module.get_candidate_read_columns(subset_name)) == expected_candidate_columns


@pytest.mark.parametrize(
    (
        "subset_name",
        "expected_origin_query",
        "expected_origin_candidate",
        "expected_canonical_query",
        "expected_canonical_candidate",
    ),
    GROUP_B_REAL_EXPECTATIONS,
)
@pytest.mark.skipif(
    not MMEB_V2_REAL_ROOT.exists(),
    reason="MMEB-V2 rerun root not available on this machine",
)
def test_group_b_image_subsets_origin_and_canonical_match_expected_first_sample(
    subset_name: str,
    expected_origin_query: str,
    expected_origin_candidate: str,
    expected_canonical_query: str,
    expected_canonical_candidate: str,
) -> None:
    dataset_path = MMEB_V2_REAL_ROOT / get_path_rel(subset_name)

    origin_dataset = build_mmeb_eval_dataset(
        dataset_name=subset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )
    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name=subset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )

    assert origin_dataset.queries[0]["query"]["text"] == expected_origin_query
    assert origin_dataset.candidates[0]["candidate"]["text"] == expected_origin_candidate
    assert canonical_dataset.queries[0]["query"]["text"] == expected_canonical_query
    assert canonical_dataset.candidates[0]["candidate"]["text"] == expected_canonical_candidate


@pytest.mark.parametrize(
    ("subset_name", "query_text", "candidate_text"),
    [
        (
            "EDIS",
            "<|image_1|>\nFind a news image that matches the provided caption: Former Israeli president and prime minister Shimon Peres right and Palestinian President Mahmoud Abbas arrive at the World Economic Forum in Southern Shuneh Jordan in May 2015.\n",
            "<|image_1|>\nRepresent the given image with related text information: World\n",
        ),
        (
            "WebQA",
            "<|image_1|>\nFind a Wikipedia image that answers this question: Does a Minnetonka Rhododendron flower have petals in a cup shape?\n",
            "<|image_1|>\nRepresent the given Wikipedia image with related text information: 2020-05-08 15 17 05 Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia Minnetonka Rhododendron flower along Tranquility Court in the Franklin Farm section of Oak Hill, Fairfax County, Virginia.\n",
        ),
    ],
)
def test_group_b_text_only_queries_apply_visual_token_alignment_in_origin_and_canonical(
    tmp_path, subset_name: str, query_text: str, candidate_text: str
) -> None:
    dataset_path = tmp_path / subset_name
    _write_eval_root(
        dataset_path,
        queries=[
            {
                "id": 0,
                "qry_text": query_text,
            }
        ],
        candidates=[
            {
                "id": 0,
                "tgt_inst": None,
                "tgt_text": candidate_text,
                "image": _make_jpeg_bytes(),
            }
        ],
        qrels=[
            {
                "query_id": 0,
                "mode": "sparse",
                "candidate_ids": [0],
                "candidate_scores": [1.0],
            }
        ],
        metadata_name=subset_name,
        task_type="image_retrieval",
    )

    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name=subset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = build_mmeb_eval_dataset(
        dataset_name=subset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )

    canonical_query = canonical_dataset.queries[0]
    canonical_candidate = canonical_dataset.candidates[0]
    origin_query = origin_dataset.queries[0]
    origin_candidate = origin_dataset.candidates[0]

    aligned_query_text = query_text.replace("<|image_1|>", "").lstrip("\n")
    assert origin_query["query"]["text"] == aligned_query_text
    assert origin_query["query"]["media"] == []
    assert canonical_query["query"]["text"] == aligned_query_text
    assert canonical_query["query"]["media"] == []
    assert origin_query.get("_validation") is None
    assert canonical_query.get("_validation") is None
    assert origin_query.get("metadata", {}).get("surface_repairs") == [
        "visual_token_alignment_strip_without_media"
    ]
    assert canonical_query.get("metadata", {}).get("surface_repairs") == [
        "visual_token_alignment_strip_without_media"
    ]
    assert origin_candidate["candidate"]["text"].startswith(STANDARD_IMAGE_TOKEN)
    assert origin_candidate["candidate"]["text"].endswith("\n")
    assert len(origin_candidate["candidate"]["media"]) == 1
    assert canonical_candidate["candidate"]["text"] == origin_candidate["candidate"]["text"]
    assert len(canonical_candidate["candidate"]["media"]) == 1


def test_group_b_visdial_canonical_dialog_query_uses_newline_after_instruction(tmp_path) -> None:
    dataset_path = tmp_path / "VisDial"
    _write_eval_root(
        dataset_path,
        queries=[
            {
                "id": 0,
                "qry_inst": None,
                "qry_text": (
                    "Represent the given dialogue about an image, which is used for image retrieval: "
                    "Q:is the photo in color\nA:yes\n"
                ),
            }
        ],
        candidates=[
            {
                "id": 0,
                "tgt_inst": None,
                "tgt_text": "<|image_1|>\nRepresent the given image.\n",
                "tgt_img_path": "VisDial/image_0.jpg",
                "image": _make_jpeg_bytes(),
            }
        ],
        qrels=[
            {
                "query_id": 0,
                "mode": "sparse",
                "candidate_ids": [0],
                "candidate_scores": [1.0],
            }
        ],
        metadata_name="VisDial",
        task_type="image_retrieval",
    )

    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name="VisDial",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = build_mmeb_eval_dataset(
        dataset_name="VisDial",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.queries[0]["query"]["text"] == (
        "Represent the given dialogue about an image, which is used for image retrieval:\n"
        "Q:is the photo in color\nA:yes\n"
    )
    assert origin_dataset.queries[0]["query"]["text"] == (
        "Represent the given dialogue about an image, which is used for image retrieval: "
        "Q:is the photo in color\nA:yes\n"
    )


def test_group_b_visdial_canonical_candidate_enforces_block_layout(tmp_path) -> None:
    dataset_path = tmp_path / "VisDial"
    _write_eval_root(
        dataset_path,
        queries=[
            {
                "id": 0,
                "qry_inst": None,
                "qry_text": "Question",
            }
        ],
        candidates=[
            {
                "id": 0,
                "tgt_inst": None,
                "tgt_text": "<|image_1|> Represent the given image.\n",
                "tgt_img_path": "VisDial/image_0.jpg",
                "image": _make_jpeg_bytes(),
            }
        ],
        qrels=[
            {
                "query_id": 0,
                "mode": "sparse",
                "candidate_ids": [0],
                "candidate_scores": [1.0],
            }
        ],
        metadata_name="VisDial",
        task_type="image_retrieval",
    )

    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name="VisDial",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )
    origin_dataset = build_mmeb_eval_dataset(
        dataset_name="VisDial",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )

    assert canonical_dataset.candidates[0]["candidate"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN}\nRepresent the given image.\n"
    )
    assert origin_dataset.candidates[0]["candidate"]["text"] == (
        f"{STANDARD_IMAGE_TOKEN} Represent the given image.\n"
    )


def test_group_b_refcoco_matching_canonical_qrels_sample_is_sparse_and_single_positive(tmp_path) -> None:
    dataset_path = tmp_path / "RefCOCO-Matching"
    _write_eval_root(
        dataset_path,
        queries=[
            {
                "id": 0,
                "qry_text": "<|image_1|>\nSelect the portion of the image that follows the language expressions: \"bowl behind the others can only see part\"\n",
                "image": _make_jpeg_bytes((0, 255, 0)),
            }
        ],
        candidates=[
            {
                "id": 0,
                "tgt_text": "<|image_1|>\nSelect the portion of the image that follows the language expressions: \"Dish in top right corner\"\n",
                "image": _make_jpeg_bytes((0, 255, 0)),
            }
        ],
        qrels=[
            {
                "query_id": 0,
                "mode": "sparse",
                "candidate_ids": [0],
                "candidate_scores": [1.0],
            }
        ],
        metadata_name="RefCOCO-Matching",
        task_type="image_visual_grounding",
    )

    dataset = build_mmeb_eval_dataset(
        dataset_name="RefCOCO-Matching",
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )

    query = dataset.queries[0]
    candidate = dataset.candidates[0]
    qrels_row = dataset.qrels.to_table().to_pylist()[0]

    assert query["query"]["media"][0]["kind"] == "image"
    assert candidate["candidate"]["media"][0]["kind"] == "image"
    assert qrels_row["mode"] == "sparse"
    assert qrels_row["candidate_ids"] == [0]
    assert qrels_row["candidate_scores"] == [1.0]
