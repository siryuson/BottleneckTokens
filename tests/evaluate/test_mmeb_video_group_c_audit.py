from __future__ import annotations

import os

from pathlib import Path

import lance
import pytest

from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import get_path_rel
from vlm2emb.data.datasets.mmeb_v2.dispatch import get_parser_module

MMEB_V2_REAL_ROOT = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2"))

pytestmark = pytest.mark.skipif(
    not MMEB_V2_REAL_ROOT.exists(),
    reason="MMEB-V2 rerun root not available on this machine",
)


GROUP_C_RUNTIME_EXPECTATIONS: dict[str, dict[str, tuple[str, str, int, int]]] = {
    "K700": {
        "origin": ("<|video_pad|> Recognize the category of the video content.", "lifting_hat", 1, 0),
        "canonical": ("<|video_pad|>\nRecognize the category of the video content.\n", "lifting_hat", 1, 0),
    },
    "UCF101": {
        "origin": (
            "<|video_pad|> What activities or sports are being performed by the person in the video?",
            "Knitting",
            1,
            0,
        ),
        "canonical": (
            "<|video_pad|>\nWhat activities or sports are being performed by the person in the video?\n",
            "Knitting",
            1,
            0,
        ),
    },
    "HMDB51": {
        "origin": (
            "<|video_pad|> What actions or objects interactions are the person in the video doing?",
            "pick",
            1,
            0,
        ),
        "canonical": (
            "<|video_pad|>\nWhat actions or objects interactions are the person in the video doing?\n",
            "pick",
            1,
            0,
        ),
    },
    "Breakfast": {
        "origin": (
            "<|video_pad|> Recognize the breakfast type that the person is cooking in the video. ",
            "milk",
            1,
            0,
        ),
        "canonical": (
            "<|video_pad|>\nRecognize the breakfast type that the person is cooking in the video.\n",
            "milk",
            1,
            0,
        ),
    },
    "SmthSmthV2": {
        "origin": (
            "<|video_pad|> What actions or object interactions are being performed by the person in the video?",
            "holding pen",
            1,
            0,
        ),
        "canonical": (
            "<|video_pad|>\nWhat actions or object interactions are being performed by the person in the video?\n",
            "holding pen",
            1,
            0,
        ),
    },
}

GROUP_C_CLASSIFICATION_PARSER_EXPECTATIONS: dict[str, dict[str, str]] = {
    "K700": {
        "origin": "<|video_pad|> Recognize the category of the video content.",
        "canonical": "<|video_pad|>\nRecognize the category of the video content.\n",
    },
    "UCF101": {
        "origin": "<|video_pad|> What activities or sports are being performed by the person in the video?",
        "canonical": "<|video_pad|>\nWhat activities or sports are being performed by the person in the video?\n",
    },
    "HMDB51": {
        "origin": "<|video_pad|> What actions or objects interactions are the person in the video doing?",
        "canonical": "<|video_pad|>\nWhat actions or objects interactions are the person in the video doing?\n",
    },
    "Breakfast": {
        "origin": "<|video_pad|> Recognize the breakfast type that the person is cooking in the video. ",
        "canonical": "<|video_pad|>\nRecognize the breakfast type that the person is cooking in the video.\n",
    },
    "SmthSmthV2": {
        "origin": "<|video_pad|> What actions or object interactions are being performed by the person in the video?",
        "canonical": "<|video_pad|>\nWhat actions or object interactions are being performed by the person in the video?\n",
    },
}

GROUP_C_RETRIEVAL_PARSER_EXPECTATIONS: dict[str, dict[str, tuple[str, str, int, int]]] = {
    "DiDeMo": {
        "origin": (
            "Find a video that includes the following described scenes: we pan right off the mural and on to a building. the view changes from artwork to a building in this shot. the screen moves to something dark the camera pans from the end of the mural to another structure and then a building.",
            "<|video_pad|> Understand the content of the provided video.",
            0,
            1,
        ),
        "canonical": (
            "Find a video that includes the following described scenes: we pan right off the mural and on to a building. the view changes from artwork to a building in this shot. the screen moves to something dark the camera pans from the end of the mural to another structure and then a building.\n",
            "<|video_pad|>\nUnderstand the content of the provided video.\n",
            0,
            1,
        ),
    },
    "MSVD": {
        "origin": (
            "Find the video snippet that corresponds to the given summary: two young men are playing table tennis",
            "<|video_pad|> Understand the content of the provided video.",
            0,
            1,
        ),
        "canonical": (
            "Find the video snippet that corresponds to the given summary: two young men are playing table tennis\n",
            "<|video_pad|>\nUnderstand the content of the provided video.\n",
            0,
            1,
        ),
    },
    "YouCook2": {
        "origin": (
            "Find a video that demonstrates the following action while making a recipe: pick the ends off the verdalago",
            "<|video_pad|> Understand the content of the provided video.",
            0,
            1,
        ),
        "canonical": (
            "Find a video that demonstrates the following action while making a recipe: pick the ends off the verdalago\n",
            "<|video_pad|>\nUnderstand the content of the provided video.\n",
            0,
            1,
        ),
    },
    "VATEX": {
        "origin": (
            "Select a video that fits the description provided: A young man is showing the polish, water. old soft cloth and brush needed to polish shoes with.",
            "<|video_pad|> Understand the content of the provided video.",
            0,
            1,
        ),
        "canonical": (
            "Select a video that fits the description provided: A young man is showing the polish, water. old soft cloth and brush needed to polish shoes with.\n",
            "<|video_pad|>\nUnderstand the content of the provided video.\n",
            0,
            1,
        ),
    },
    "MSR-VTT": {
        "origin": (
            "Find a video that contains the following visual content: a woman creating a fondant baby and flower",
            "<|video_pad|> Understand the content of the provided video.",
            0,
            1,
        ),
        "canonical": (
            "Find a video that contains the following visual content: a woman creating a fondant baby and flower\n",
            "<|video_pad|>\nUnderstand the content of the provided video.\n",
            0,
            1,
        ),
    },
}


def _build_first_runtime_sample(dataset_name: str, mode: str) -> tuple[str, str, int, int]:
    dataset_path = MMEB_V2_REAL_ROOT / get_path_rel(dataset_name)
    dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": mode},
    )
    query = dataset.queries[0]["query"]
    candidate = dataset.candidates[0]["candidate"]
    return query["text"], candidate["text"], len(query["media"]), len(candidate["media"])


def _build_first_parser_sample(dataset_name: str, mode: str) -> tuple[str, str, int, int]:
    dataset_path = MMEB_V2_REAL_ROOT / get_path_rel(dataset_name)
    parser_module = get_parser_module(dataset_name)

    query_raw = lance.dataset(dataset_path / "queries.lance").to_table(
        columns=list(parser_module.get_query_read_columns(dataset_name)),
        limit=1,
    ).to_pylist()[0]
    candidate_raw = lance.dataset(dataset_path / "candidates.lance").to_table(
        columns=list(parser_module.get_candidate_read_columns(dataset_name)),
        limit=1,
    ).to_pylist()[0]

    query_item = parser_module.build_query_transform(
        dataset_name=dataset_name,
        transform_kwargs={"runtime_mode": mode},
    )(query_raw)
    candidate_item = parser_module.build_candidate_transform(
        dataset_name=dataset_name,
        transform_kwargs={"runtime_mode": mode},
    )(candidate_raw)
    query = query_item["query"]
    candidate = candidate_item["candidate"]
    return query["text"], candidate["text"], len(query["media"]), len(candidate["media"])


def _build_first_query_via_parser(dataset_name: str, mode: str) -> str:
    dataset_path = MMEB_V2_REAL_ROOT / get_path_rel(dataset_name)
    parser_module = get_parser_module(dataset_name)
    query_raw = lance.dataset(dataset_path / "queries.lance").to_table(
        columns=list(parser_module.get_query_read_columns(dataset_name)),
        limit=1,
    ).to_pylist()[0]
    query_item = parser_module.build_query_transform(
        dataset_name=dataset_name,
        transform_kwargs={"runtime_mode": mode},
    )(query_raw)
    return query_item["query"]["text"]


def test_group_c_runtime_first_samples_match_archive_and_current_contract():
    for dataset_name, expectations in GROUP_C_RUNTIME_EXPECTATIONS.items():
        assert _build_first_runtime_sample(dataset_name, "origin") == expectations["origin"]
        assert _build_first_runtime_sample(dataset_name, "canonical") == expectations["canonical"]


def test_group_c_video_classification_parsers_own_mode_specific_query_newline():
    for dataset_name, expectations in GROUP_C_CLASSIFICATION_PARSER_EXPECTATIONS.items():
        assert _build_first_query_via_parser(dataset_name, "origin") == expectations["origin"]
        assert _build_first_query_via_parser(dataset_name, "canonical") == expectations["canonical"]


def test_group_c_video_retrieval_parsers_own_canonical_newline_policy():
    for dataset_name, expectations in GROUP_C_RETRIEVAL_PARSER_EXPECTATIONS.items():
        assert _build_first_parser_sample(dataset_name, "origin") == expectations["origin"]
        assert _build_first_parser_sample(dataset_name, "canonical") == expectations["canonical"]
