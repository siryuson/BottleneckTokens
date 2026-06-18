from __future__ import annotations

import os

from pathlib import Path

import pytest

from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import get_path_rel

MMEB_V2_REAL_ROOT = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2"))

pytestmark = pytest.mark.skipif(
    not MMEB_V2_REAL_ROOT.exists(),
    reason="MMEB-V2 rerun root not available on this machine",
)

GROUP_A_EXPECTATIONS: list[tuple[str, str, str]] = [
    (
        "HatefulMemes",
        "<|image_pad|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not\n",
        "Yes",
    ),
    (
        "VOC2007",
        "<|image_pad|>\nIdentify the object shown in the image\n",
        "chair",
    ),
    (
        "SUN397",
        "<|image_pad|>\nIdentify the scene shown in the image\n",
        "firing range indoor",
    ),
    (
        "Place365",
        "<|image_pad|>\nIdentify the scene shown in the image\n",
        "berth",
    ),
    (
        "ImageNet-A",
        "<|image_pad|>\nRepresent the given image for classification\n",
        "Rottweiler",
    ),
    (
        "ImageNet-R",
        "<|image_pad|>\nRepresent the given image for classification\n",
        "flute",
    ),
    (
        "ObjectNet",
        "<|image_pad|>\nIdentify the object shown in the image\n",
        "tray",
    ),
    (
        "Country211",
        "<|image_pad|>\nIdentify the country depicted in the image\n",
        "Fiji",
    ),
    (
        "N24News",
        "<|image_pad|>\nRepresent the given news image with the following caption for domain classification: Anthony Pidgeon\n",
        "Travel",
    ),
    (
        "ImageNet-1K",
        "<|image_pad|>\nRepresent the given image for classification\n",
        "coucal",
    ),
    (
        "VizWiz",
        "<|image_pad|>\nRepresent the given image with the following question: Can you tell me what this medicine is please?\n",
        "night time",
    ),
    (
        "ChartQA",
        "<|image_pad|>\nRepresent the given image with the following question: How many food item is shown in the bar graph?\n",
        "14",
    ),
    (
        "ScienceQA",
        "<|image_pad|>\nRepresent the given image with the following question: Which of the following could Gordon's test show?\n",
        "how steady a parachute with a 1 m vent was at 200 km per hour",
    ),
    (
        "GQA",
        "<|image_pad|>\nRepresent the given image with the following question: What is this bird called?\n",
        "This is a parrot.",
    ),
    (
        "A-OKVQA",
        "<|image_pad|>\nRepresent the given image with the following question: What is in the motorcyclist's mouth?\n",
        "cigarette",
    ),
    (
        "DocVQA",
        "<|image_pad|>\nRepresent the given image with the following question: What is the ‘actual’ value per 1000, during the year 1975?\n",
        "0.28",
    ),
    (
        "InfographicsVQA",
        "<|image_pad|>\nRepresent the given image with the following question: Which social platform has heavy female audience?\n",
        "pinterest",
    ),
    (
        "Visual7W",
        "<|image_pad|>\nRepresent the given image with the following question: What color is the sidewalk?\n",
        "Gray.",
    ),
    (
        "TextVQA",
        "<|image_pad|>\nRepresent the given image with the following question: what is the brand of this camera?\n",
        "dakota",
    ),
    (
        "OK-VQA",
        "<|image_pad|>\nRepresent the given image with the following question: What sport can you use this for?\n",
        "race",
    ),
]


@pytest.mark.parametrize(("subset_name", "expected_query_text", "expected_candidate_text"), GROUP_A_EXPECTATIONS)
def test_group_a_image_subsets_origin_and_canonical_match_archive_first_sample(
    subset_name: str,
    expected_query_text: str,
    expected_candidate_text: str,
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

    origin_query = origin_dataset.queries[0]["query"]["text"]
    origin_candidate = origin_dataset.candidates[0]["candidate"]["text"]
    canonical_query = canonical_dataset.queries[0]["query"]["text"]
    canonical_candidate = canonical_dataset.candidates[0]["candidate"]["text"]

    assert origin_query == expected_query_text
    assert origin_candidate == expected_candidate_text
    assert canonical_query == expected_query_text
    assert canonical_candidate == expected_candidate_text
    assert origin_query == canonical_query
    assert origin_candidate == canonical_candidate
