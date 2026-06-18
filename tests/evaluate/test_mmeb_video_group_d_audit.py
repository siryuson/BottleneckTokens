from __future__ import annotations

import os

from pathlib import Path

import pytest

from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset

MMEB_V2_REAL_ROOT = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2"))

pytestmark = pytest.mark.skipif(
    not MMEB_V2_REAL_ROOT.exists(),
    reason="MMEB-V2 rerun root not available on this machine",
)


GROUP_D_CASES = [
    (
        "Video-MME",
        "video_question_answer",
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  When demonstrating the Germany modern Christmas tree "
        "is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?\n"
        "A. Apples.\nB. Candles.\nC. Berries.\nD. The three kinds are of the same number.",
        "A. Apples.",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  When demonstrating the Germany modern Christmas tree "
        "is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?\n"
        "A. Apples.\nB. Candles.\nC. Berries.\nD. The three kinds are of the same number.\n",
        "A. Apples.",
    ),
    (
        "NExTQA",
        "video_question_answer",
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  what did the baby do after throwing the green cup "
        "away while on the floor near the end\nOptions:\n(A) clap proudly\n(B) the lady sitting down\n(C) lay on floor\n"
        "(D) just picked it up\n(E) crawl",
        "clap proudly",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  what did the baby do after throwing the green cup "
        "away while on the floor near the end\nOptions:\n(A) clap proudly\n(B) the lady sitting down\n(C) lay on floor\n"
        "(D) just picked it up\n(E) crawl\n",
        "clap proudly",
    ),
    (
        "EgoSchema",
        "video_question_answer",
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  Taking into account all the actions performed by c, "
        "what can you deduce about the primary objective and focus within the video content?A. C is cooking. B. C is doing "
        "laundry. C. C is cleaning the kitchen. D. C is cleaning dishes. E. C is cleaning the bathroom.",
        "A. C is cooking.",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  Taking into account all the actions performed by c, "
        "what can you deduce about the primary objective and focus within the video content?A. C is cooking. B. C is doing "
        "laundry. C. C is cleaning the kitchen. D. C is cleaning dishes. E. C is cleaning the bathroom.\n",
        "A. C is cooking.",
    ),
    (
        "MVBench",
        "video_question_answer",
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  Why did Castle dress like a fairy when he was "
        "speaking to Emily?\nOptions:\n(A) To get her to trust him\n(B) He secretly loved fairies\n(C) He lost a bet with "
        "Emily\n(D) It was dress like a fairy day at school\n(E) Mrs Ruiz made him dress up",
        "(A) To get her to trust him",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  Why did Castle dress like a fairy when he was "
        "speaking to Emily?\nOptions:\n(A) To get her to trust him\n(B) He secretly loved fairies\n(C) He lost a bet with "
        "Emily\n(D) It was dress like a fairy day at school\n(E) Mrs Ruiz made him dress up\n",
        "(A) To get her to trust him",
    ),
    (
        "ActivityNetQA",
        "video_question_answer",
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  does the girl in black clothes have long hair? "
        "(A) yes; (B) no.",
        "yes",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  does the girl in black clothes have long hair? "
        "(A) yes; (B) no.\n",
        "yes",
    ),
]

QUERY_NORMALIZATION_CASES = [
    (
        "MVBench",
        22,
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  What was Amy and Rachel doing before Ross came in "
        "the room. \nOptions:\n(A) Amy and Rachel where cooking before Ross came in the room.\n(B) Amy and Rachel where "
        "fighting when Ross came in the room.\n(C) Amy and Rachel Where getting dressed when Ross came in the room.\n"
        "(D) Amy and Rachel was talking before Ross came in the room.\n(E) Amy and Rachel was cleaning before Ross came in "
        "the room.",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  What was Amy and Rachel doing before Ross came in "
        "the room.\nOptions:\n(A) Amy and Rachel where cooking before Ross came in the room.\n(B) Amy and Rachel where "
        "fighting when Ross came in the room.\n(C) Amy and Rachel Where getting dressed when Ross came in the room.\n"
        "(D) Amy and Rachel was talking before Ross came in the room.\n(E) Amy and Rachel was cleaning before Ross came in "
        "the room.\n",
    ),
    (
        "NExTQA",
        334,
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  why does the man in red turn the camera to himself "
        "in the middle of the video\nOptions:\n(A) look at phone\n(B) to talk to it\n(C) to check his face\n"
        "(D) to fix the camera \n(E) blow dust off lenses",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  why does the man in red turn the camera to himself "
        "in the middle of the video\nOptions:\n(A) look at phone\n(B) to talk to it\n(C) to check his face\n"
        "(D) to fix the camera\n(E) blow dust off lenses\n",
    ),
    (
        "ActivityNetQA",
        101,
        "<|video_pad|> Given a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  does this sport need leg strength ? (A) yes; (B) no.",
        "<|video_pad|>\nGiven a video and a question, select the most accurate answer from the provided candidates. "
        "Return only the exact text of your chosen answer. Question:  does this sport need leg strength? (A) yes; (B) no.\n",
    ),
]

CANDIDATE_NORMALIZATION_CASES = [
    ("MVBench", 5, "(A) She is jealous about Bernadette's looks. ", "(A) She is jealous about Bernadette's looks."),
    ("NExTQA", 1673, "to fix the camera ", "to fix the camera"),
]


@pytest.mark.parametrize(
    "dataset_name,runtime_type,expected_origin_query,expected_origin_candidate,expected_canonical_query,expected_canonical_candidate",
    GROUP_D_CASES,
)
def test_group_d_video_qa_origin_and_canonical_runtime_contract(
    dataset_name,
    runtime_type,
    expected_origin_query,
    expected_origin_candidate,
    expected_canonical_query,
    expected_canonical_candidate,
):
    dataset_path = MMEB_V2_REAL_ROOT / "video-tasks" / dataset_name

    origin_dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )
    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )

    origin_query = origin_dataset.queries[0]["query"]["text"]
    origin_candidate = origin_dataset.candidates[0]["candidate"]["text"]
    canonical_query = canonical_dataset.queries[0]["query"]["text"]
    canonical_candidate = canonical_dataset.candidates[0]["candidate"]["text"]

    assert origin_query == expected_origin_query
    assert origin_candidate == expected_origin_candidate
    assert canonical_query == expected_canonical_query
    assert canonical_candidate == expected_canonical_candidate


@pytest.mark.parametrize("dataset_name,index,expected_origin_query,expected_canonical_query", QUERY_NORMALIZATION_CASES)
def test_group_d_video_qa_query_whitespace_normalization(dataset_name, index, expected_origin_query, expected_canonical_query):
    dataset_path = MMEB_V2_REAL_ROOT / "video-tasks" / dataset_name

    origin_dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )
    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )

    assert origin_dataset.queries[index]["query"]["text"] == expected_origin_query
    assert canonical_dataset.queries[index]["query"]["text"] == expected_canonical_query


@pytest.mark.parametrize("dataset_name,index,expected_origin_candidate,expected_canonical_candidate", CANDIDATE_NORMALIZATION_CASES)
def test_group_d_video_qa_candidate_whitespace_normalization(
    dataset_name,
    index,
    expected_origin_candidate,
    expected_canonical_candidate,
):
    dataset_path = MMEB_V2_REAL_ROOT / "video-tasks" / dataset_name

    origin_dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "origin"},
    )
    canonical_dataset = build_mmeb_eval_dataset(
        dataset_name=dataset_name,
        artifact_path=str(dataset_path),
        transform_kwargs={"runtime_mode": "canonical"},
    )

    assert origin_dataset.candidates[index]["candidate"]["text"] == expected_origin_candidate
    assert canonical_dataset.candidates[index]["candidate"]["text"] == expected_canonical_candidate
