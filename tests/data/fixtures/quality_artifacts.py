"""Fixture builders for Phase 2 quality validation tests."""

from __future__ import annotations

from vlm2emb.data.quality.models import QualityFinding, ValidationResult
from vlm2emb.data.quality.reason_codes import BROKEN_MEDIA, EMPTY_TEXT, INVALID_POSITIVE


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc```\xf8\x0f\x00\x01\x01\x01\x00"
    b"\x18\xdd\x8d\xb1"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def make_retrieval_artifacts() -> tuple[list[dict], list[dict], list[dict]]:
    """Build one minimal canonical retrieval artifact triple."""

    queries = [{
        "id": "query-1",
        "sample_id": "sample_query_1",
        "asset_ids": ["asset_query_1"],
        "text": "describe the image",
        "images": [PNG_BYTES],
    }]
    candidates = [{
        "id": "candidate-1",
        "sample_id": "sample_candidate_1",
        "asset_ids": ["asset_candidate_1"],
        "text": "tabby cat",
        "images": [PNG_BYTES],
    }]
    qrels = [{
        "relation_id": "relation_qrels_1",
        "query_id": "query-1",
        "mode": "image",
        "candidate_ids": ["candidate-1"],
        "candidate_scores": [1.0],
    }]
    return queries, candidates, qrels


def make_validation_result() -> ValidationResult:
    """Build one representative validation result for cleaning/reporting tests."""

    findings = (
        QualityFinding(
            reason_code=EMPTY_TEXT,
            severity="error",
            scope="sample",
            decision_hint="drop",
            dataset_name="demo",
            message="Empty text payload in query",
            sample_id="sample_query_1",
        ),
        QualityFinding(
            reason_code=BROKEN_MEDIA,
            severity="error",
            scope="asset",
            decision_hint="quarantine",
            dataset_name="demo",
            message="Broken image bytes",
            sample_id="sample_candidate_1",
            asset_id="asset_candidate_1",
        ),
        QualityFinding(
            reason_code=INVALID_POSITIVE,
            severity="error",
            scope="relation",
            decision_hint="block",
            dataset_name="demo",
            message="No positive candidate",
            relation_id="relation_qrels_1",
        ),
    )
    return ValidationResult(dataset_name="demo", findings=findings)
