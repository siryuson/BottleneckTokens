from __future__ import annotations

import os
from pathlib import Path

import pytest

from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset

# Audit visdoc Group F parser contracts against the shared rule vocabulary:
# - origin query: trailing_newline off
# - canonical query: trailing_newline on
# - origin candidate: visual_token_separator=space, trailing_newline off
# - canonical candidate: visual_token_separator=newline, trailing_newline on


VISDOC_GROUP_F_SUBSETS = [
    "MMLongBench-page",
    "MMLongBench-doc",
    "ViDoSeek-doc",
    "ViDoSeek-page",
    "ViDoRe_syntheticDocQA_artificial_intelligence",
    "ViDoRe_syntheticDocQA_government_reports",
    "ViDoRe_syntheticDocQA_healthcare_industry",
    "ViDoRe_shiftproject",
    "ViDoRe_syntheticDocQA_energy",
    "ViDoRe_infovqa",
    "ViDoRe_tabfquad",
    "ViDoRe_tatdqa",
    "ViDoRe_arxivqa",
    "ViDoRe_docvqa",
    "ViDoRe_esg_reports_human_labeled_v2",
    "ViDoRe_biomedical_lectures_v2_multilingual",
    "ViDoRe_economics_reports_v2_multilingual",
    "ViDoRe_esg_reports_v2_multilingual",
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
    "VisRAG_InfoVQA",
    "VisRAG_PlotQA",
]

QUERY_PREFIX = "Find a document image that matches the given query: "
# visual_token_separator=space, trailing_newline=off
ORIGIN_CANDIDATE_TEXT = "<|image_pad|> Understand the content of the provided document image."
# visual_token_separator=newline, trailing_newline=on
CANONICAL_CANDIDATE_TEXT = "<|image_pad|>\nUnderstand the content of the provided document image.\n"
PAGE_INDEX_SUBSETS = {
    "MMLongBench-page",
    "MMLongBench-doc",
    "ViDoSeek-doc",
    "ViDoSeek-page",
}
ORIGIN_QUERY_PRESERVE_SUBSETS = {
    # These subsets are tracked as origin-preserve exceptions for query tail
    # shape. The source query can be mixed-tail, so origin should not enforce
    # a global trailing_newline=off assertion here.
    "MMLongBench-page",
    "MMLongBench-doc",
}
PATH_SUBSETS = {
    "MMLongBench-page",
    "MMLongBench-doc",
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
    "VisRAG_InfoVQA",
    "VisRAG_PlotQA",
    "ViDoSeek-page",
}


def _build_sample(subset_name: str, runtime_mode: str) -> tuple[dict[str, object], dict[str, object]]:
    root = Path(os.environ.get("VLM2EMB_MMEB_V2_ROOT", "./data/MMEB-V2")) / "visdoc-tasks"
    if not root.exists():
        pytest.skip(f"MMEB-V2 visdoc Lance root not found: {root}")
    dataset = build_mmeb_eval_dataset(
        dataset_name=subset_name,
        artifact_path=str(root / subset_name),
        transform_kwargs={"runtime_mode": runtime_mode},
    )
    return dataset.queries[0], dataset.candidates[0]


@pytest.mark.parametrize("subset_name", VISDOC_GROUP_F_SUBSETS)
def test_visdoc_group_f_origin_and_canonical_samples_match_contract(subset_name: str) -> None:
    origin_query, origin_candidate = _build_sample(subset_name, "origin")
    canonical_query, canonical_candidate = _build_sample(subset_name, "canonical")

    assert origin_query["query"]["text"].startswith(QUERY_PREFIX)
    if subset_name not in ORIGIN_QUERY_PRESERVE_SUBSETS:
        assert not origin_query["query"]["text"].endswith("\n")
    assert canonical_query["query"]["text"].startswith(QUERY_PREFIX)
    assert canonical_query["query"]["text"].endswith("\n")

    assert origin_candidate["candidate"]["text"] == ORIGIN_CANDIDATE_TEXT
    assert canonical_candidate["candidate"]["text"] == CANONICAL_CANDIDATE_TEXT

    assert origin_candidate["candidate"]["media"][0]["kind"] == "image"
    assert canonical_candidate["candidate"]["media"][0]["kind"] == "image"

    if subset_name in PATH_SUBSETS:
        assert origin_candidate["metadata"]["path"]
        assert canonical_candidate["metadata"]["path"]
        assert origin_candidate["metadata"]["path"] == canonical_candidate["metadata"]["path"]
    else:
        assert "metadata" not in origin_candidate
        assert "metadata" not in canonical_candidate

    if subset_name in PAGE_INDEX_SUBSETS:
        assert "corpus_range" in origin_query["metadata"]
        assert "corpus_range" in canonical_query["metadata"]
        assert origin_query["metadata"]["corpus_range"] == canonical_query["metadata"]["corpus_range"]
        assert isinstance(origin_query["metadata"]["corpus_range"], list)
        assert origin_query["metadata"]["corpus_range"]
    else:
        assert "metadata" not in origin_query
        assert "metadata" not in canonical_query
