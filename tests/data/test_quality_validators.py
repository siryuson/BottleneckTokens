from __future__ import annotations

import json

import lance
import pyarrow as pa

from vlm2emb.data.quality import (
    BROKEN_MEDIA,
    EMPTY_QRELS,
    EMPTY_TEXT,
    ID_COLLISION,
    INVALID_IDENTIFIER,
    MISSING_CANDIDATE_LINK,
    MISSING_QUERY_LINK,
    MISSING_REFERENCE,
    REASON_CODE_REGISTRY,
    RELATION_CARDINALITY_MISMATCH,
    SCHEMA_MODALITY_CONSTRAINT_FAILED,
    SCHEMA_REQUIRED_FIELD_MISSING,
    ValidationResult,
)
from vlm2emb.data.quality.gates import validate_document_train_dataset
from vlm2emb.data.quality.validators import (
    validate_id_uniqueness,
    validate_media_payloads,
    validate_modality_constraints,
    validate_reference_links,
    validate_required_fields,
    validate_retrieval_integrity,
    validate_text_not_empty,
)

from tests.data.fixtures.quality_artifacts import make_retrieval_artifacts


def test_quality_contract_exposes_blocking_result_summary():
    result = ValidationResult(dataset_name="demo")
    assert not result.has_blocking_errors
    assert result.findings_by_reason == {}


def test_reason_code_registry_covers_phase2_core_problem_families():
    expected_codes = {
        SCHEMA_REQUIRED_FIELD_MISSING,
        SCHEMA_MODALITY_CONSTRAINT_FAILED,
        EMPTY_TEXT,
        BROKEN_MEDIA,
        MISSING_REFERENCE,
        ID_COLLISION,
        EMPTY_QRELS,
        MISSING_QUERY_LINK,
        MISSING_CANDIDATE_LINK,
        INVALID_IDENTIFIER,
        RELATION_CARDINALITY_MISMATCH,
    }
    assert expected_codes.issubset(REASON_CODE_REGISTRY)


def test_quality_fixture_builds_minimal_retrieval_artifacts():
    queries, candidates, qrels = make_retrieval_artifacts()
    assert queries[0]["sample_id"] == "sample_query_1"
    assert candidates[0]["asset_ids"] == ["asset_candidate_1"]
    assert qrels[0]["relation_id"] == "relation_qrels_1"


def test_structural_validator_flags_required_fields_and_id_collisions():
    rows = [
        {"sample_id": "sample-1", "text": "", "id": "dup"},
        {"sample_id": "sample-2", "text": "ok", "id": "dup"},
    ]
    findings = validate_required_fields(
        rows,
        dataset_name="demo",
        required_fields=("text",),
    )
    findings += validate_id_uniqueness(rows, dataset_name="demo", id_field="id")
    assert {finding.reason_code for finding in findings} == {
        SCHEMA_REQUIRED_FIELD_MISSING,
        ID_COLLISION,
    }


def test_media_validator_flags_empty_text_and_broken_media():
    rows = [{
        "sample_id": "sample-1",
        "asset_ids": ["asset-1"],
        "text": "",
        "images": [b"not-a-real-image"],
    }]
    findings = validate_text_not_empty(rows, dataset_name="demo")
    findings += validate_media_payloads(rows, dataset_name="demo")
    assert {finding.reason_code for finding in findings} == {EMPTY_TEXT, BROKEN_MEDIA}


def test_structural_validator_flags_modality_constraints_and_missing_links():
    rows = [{"sample_id": "sample-1", "text": "", "images": []}]
    findings = validate_modality_constraints(rows, dataset_name="demo")
    findings += validate_reference_links(
        [{"relation_id": "relation-1", "candidate_ids": ["missing-candidate"]}],
        dataset_name="demo",
        field_name="candidate_ids",
        existing_ids={"candidate-1"},
    )
    assert {finding.reason_code for finding in findings} == {
        SCHEMA_MODALITY_CONSTRAINT_FAILED,
        MISSING_REFERENCE,
    }


def test_retrieval_validator_flags_empty_qrels_and_missing_links():
    queries, candidates, _ = make_retrieval_artifacts()
    findings = validate_retrieval_integrity(
        dataset_name="demo",
        queries=queries,
        candidates=candidates,
        qrels=[],
    )
    assert [finding.reason_code for finding in findings] == [EMPTY_QRELS]


def test_retrieval_validator_flags_missing_query_candidate_and_invalid_positive():
    queries, candidates, qrels = make_retrieval_artifacts()
    qrels[0]["query_id"] = "missing-query"
    qrels[0]["candidate_ids"] = ["missing-candidate"]
    qrels[0]["candidate_scores"] = [0.0]
    findings = validate_retrieval_integrity(
        dataset_name="demo",
        queries=queries,
        candidates=candidates,
        qrels=qrels,
    )
    assert {finding.reason_code for finding in findings} == {
        MISSING_QUERY_LINK,
        MISSING_CANDIDATE_LINK,
        "invalid_positive",
    }


def test_runtime_retrieval_bundle_validator_flags_bad_stable_ids_and_score_mismatch():
    queries, candidates, qrels = make_retrieval_artifacts()
    queries[0]["sample_id"] = "sample-query-1"
    candidates[0]["asset_ids"] = ["asset-candidate-1"]
    qrels[0]["relation_id"] = "relation-qrels-1"
    qrels[0]["candidate_scores"] = [1.0, 0.0]

    from vlm2emb.data.quality.validators.runtime import validate_retrieval_bundle

    result = validate_retrieval_bundle(
        dataset_name="demo",
        queries=queries,
        candidates=candidates,
        qrels=qrels,
    )

    assert {finding.reason_code for finding in result.findings} >= {
        INVALID_IDENTIFIER,
        RELATION_CARDINALITY_MISMATCH,
    }


def test_document_train_validator_accepts_valid_vidore_rag_root(tmp_path):
    dataset_path = tmp_path / "vidore-rag"
    (dataset_path / "data").mkdir(parents=True)
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                {
                    "query": "find the matching page",
                    "answer": "answer",
                    "source": "docvqa",
                    "image": {
                        "bytes": (
                            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
                            b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
                        ),
                        "path": "page.png",
                    },
                }
            ]
        ),
        str(dataset_path / "data" / "train.lance"),
        mode="create",
    )
    with (dataset_path / "manifest.json").open("w") as file_handle:
        json.dump(
            {
                "dataset_name": "vidore-rag",
                "task_type": "document_retrieval",
                "metadata": {"loader_type": "vidore_rag_train"},
            },
            file_handle,
        )

    metadata, result = validate_document_train_dataset(dataset_path, media_sample_limit=4)

    assert metadata["loader_type"] == "vidore_rag_train"
    assert result.findings == ()
    assert result.metadata["row_count"] == 1


def test_document_train_validator_flags_empty_query_and_missing_image(tmp_path):
    dataset_path = tmp_path / "visrag"
    (dataset_path / "data").mkdir(parents=True)
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                {
                    "query": "",
                    "source": "ArxivQA",
                    "image": None,
                }
            ]
        ),
        str(dataset_path / "data" / "train.lance"),
        mode="create",
    )
    with (dataset_path / "manifest.json").open("w") as file_handle:
        json.dump(
            {
                "dataset_name": "visrag",
                "task_type": "document_retrieval",
                "metadata": {"loader_type": "visrag_train"},
            },
            file_handle,
        )

    _, result = validate_document_train_dataset(dataset_path, media_sample_limit=4)

    assert {finding.reason_code for finding in result.findings} == {
        EMPTY_TEXT,
        SCHEMA_REQUIRED_FIELD_MISSING,
    }
