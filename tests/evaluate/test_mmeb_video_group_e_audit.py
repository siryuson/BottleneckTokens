from __future__ import annotations

import os

from pathlib import Path

import lance
import pytest

from vlm2emb.data.datasets.mmeb_v2.dispatch import get_parser_module

MMEB_V2_REAL_ROOT = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2"))

pytestmark = pytest.mark.skipif(
    not MMEB_V2_REAL_ROOT.exists(),
    reason="MMEB-V2 rerun root not available on this machine",
)


def _build_parser_sample(
    dataset_name: str,
    relative_path: str,
    runtime_mode: str,
    *,
    role: str,
    offset: int = 0,
):
    parser_module = get_parser_module(dataset_name)
    dataset_path = MMEB_V2_REAL_ROOT / relative_path
    if role == "query":
        columns = list(parser_module.get_query_read_columns(dataset_name))
        transform = parser_module.build_query_transform(
            dataset_name=dataset_name,
            transform_kwargs={"runtime_mode": runtime_mode},
        )
        lance_path = dataset_path / "queries.lance"
    else:
        columns = list(parser_module.get_candidate_read_columns(dataset_name))
        transform = parser_module.build_candidate_transform(
            dataset_name=dataset_name,
            transform_kwargs={"runtime_mode": runtime_mode},
        )
        lance_path = dataset_path / "candidates.lance"
    row = lance.dataset(lance_path).to_table(columns=columns, offset=offset, limit=1).to_pylist()[0]
    return transform(row)


def test_group_e_qvhighlight_and_charades_parser_contract_matches_archive_and_canonical_plan():
    cases = [
        (
            "QVHighlight",
            "video-tasks/QVHighlight",
            "<|video_pad|> Find the clip that corresponds to the described scene in the given video: A girl and her mother cooked while talking with each other on facetime.",
            "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: A girl and her mother cooked while talking with each other on facetime.\n",
        ),
        (
            "Charades-STA",
            "video-tasks/Charades-STA",
            "<|video_pad|> Find the clip that corresponds to the described scene in the given video: person turn a light on.",
            "<|video_pad|>\nFind the clip that corresponds to the described scene in the given video: person turn a light on.\n",
        ),
    ]

    for dataset_name, relative_path, expected_origin_query_text, expected_canonical_query_text in cases:
        origin_query = _build_parser_sample(dataset_name, relative_path, "origin", role="query")
        canonical_query = _build_parser_sample(dataset_name, relative_path, "canonical", role="query")
        origin_candidate = _build_parser_sample(dataset_name, relative_path, "origin", role="candidate")
        canonical_candidate = _build_parser_sample(dataset_name, relative_path, "canonical", role="candidate")

        assert origin_query["query"]["text"] == expected_origin_query_text
        assert canonical_query["query"]["text"] == expected_canonical_query_text
        assert origin_candidate["candidate"]["text"] == "<|video_pad|> Understand the content of the provided video."
        assert canonical_candidate["candidate"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video.\n"


def test_group_e_momentseeker_parser_contract_covers_text_image_video_queries():
    expected_queries = {
        "video": (
            0,
            "<|video_pad|> Find the clip that corresponds to the given sentence and video segment: What happened to this window afterward?",
            "<|video_pad|>\nFind the clip that corresponds to the given sentence and video segment: What happened to this window afterward?\n",
        ),
        "image": (
            100,
            "<|image_pad|> Select the video clip that aligns with the given text and image: Did the man and the woman in the picture embrace?",
            "<|image_pad|>\nSelect the video clip that aligns with the given text and image: Did the man and the woman in the picture embrace?\n",
        ),
        "text": (
            300,
            "Find the clip that corresponds to the given text: What game did I play after playing Connect 4?",
            "Find the clip that corresponds to the given text: What game did I play after playing Connect 4?\n",
        ),
    }

    for kind, (offset, expected_origin_query_text, expected_canonical_query_text) in expected_queries.items():
        canonical_sample = _build_parser_sample(
            "MomentSeeker",
            "video-tasks/MomentSeeker",
            "canonical",
            role="query",
            offset=offset,
        )
        origin_sample = _build_parser_sample(
            "MomentSeeker",
            "video-tasks/MomentSeeker",
            "origin",
            role="query",
            offset=offset,
        )
        assert canonical_sample["query"]["text"] == expected_canonical_query_text
        assert origin_sample["query"]["text"] == expected_origin_query_text
        if kind == "text":
            assert canonical_sample["query"]["media"] == []
            assert origin_sample["query"]["media"] == []
        else:
            assert canonical_sample["query"]["media"][0]["kind"] == kind
            assert origin_sample["query"]["media"][0]["kind"] == kind

    canonical_candidate = _build_parser_sample(
        "MomentSeeker",
        "video-tasks/MomentSeeker",
        "canonical",
        role="candidate",
    )
    origin_candidate = _build_parser_sample(
        "MomentSeeker",
        "video-tasks/MomentSeeker",
        "origin",
        role="candidate",
    )
    assert origin_candidate["candidate"]["text"] == "<|video_pad|> Understand the content of the provided video clip."
    assert canonical_candidate["candidate"]["text"] == "<|video_pad|>\nUnderstand the content of the provided video clip.\n"


def test_group_e_candidate_shape_audit_confirms_fixed_parser_owned_templates():
    qvhighlight_candidates = lance.dataset(
        MMEB_V2_REAL_ROOT / "video-tasks/QVHighlight/candidates.lance"
    )
    charades_candidates = lance.dataset(
        MMEB_V2_REAL_ROOT / "video-tasks/Charades-STA/candidates.lance"
    )
    momentseeker_candidates = lance.dataset(
        MMEB_V2_REAL_ROOT / "video-tasks/MomentSeeker/candidates.lance"
    )

    for dataset in (qvhighlight_candidates, charades_candidates):
        total_rows = dataset.count_rows()
        for offset in (0, total_rows // 2, total_rows - 1):
            row = dataset.to_table(offset=offset, limit=1).to_pylist()[0]
            assert sorted(row.keys()) == ["id", "video"]
            assert isinstance(row["video"], list)
            assert row["video"]

    total_rows = momentseeker_candidates.count_rows()
    for offset in (0, total_rows // 2, total_rows - 1):
        row = momentseeker_candidates.to_table(offset=offset, limit=1).to_pylist()[0]
        assert sorted(row.keys()) == ["end_time", "frame_path", "id", "output_path", "start_time", "video"]
        assert isinstance(row["video"], list)
        assert row["video"]

    metadata_table = momentseeker_candidates.to_table(
        columns=["start_time", "end_time", "frame_path", "output_path"]
    ).to_pylist()
    assert metadata_table
    assert all(row["start_time"] is not None for row in metadata_table)
    assert all(row["end_time"] is not None for row in metadata_table)
    assert all(isinstance(row["frame_path"], str) and row["frame_path"] for row in metadata_table)
    assert all(isinstance(row["output_path"], str) and row["output_path"] for row in metadata_table)
