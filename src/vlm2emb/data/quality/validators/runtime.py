"""Runtime-oriented validators built on top of the shared quality core."""

from __future__ import annotations

from typing import Any

from ..models import ValidationResult
from ..reason_codes import INVALID_IDENTIFIER, RELATION_CARDINALITY_MISMATCH
from .media import validate_media_payloads
from .retrieval import _build_finding
from .retrieval import validate_retrieval_integrity
from .structural import validate_id_uniqueness, validate_modality_constraints, validate_required_fields


def _validate_prefixed_identifier(
    *,
    dataset_name: str,
    rows: list[dict[str, Any]],
    field_name: str,
    expected_prefix: str,
    relation_scope: bool = False,
) -> list:
    findings = []
    for row in rows:
        value = row.get(field_name)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            item_value = str(item)
            if item_value and not item_value.startswith(expected_prefix):
                findings.append(
                    _build_finding(
                        INVALID_IDENTIFIER,
                        dataset_name=dataset_name,
                        relation_id=str(row.get("relation_id")) if relation_scope and row.get("relation_id") else None,
                        message=(
                            f"Field {field_name!r} must start with {expected_prefix!r}, "
                            f"got {item_value!r}."
                        ),
                        metadata={"field_name": field_name, "value": item_value, "expected_prefix": expected_prefix},
                    )
                )
    return findings


def _validate_relation_cardinality(
    *,
    dataset_name: str,
    qrels: list[dict[str, Any]],
) -> list:
    findings = []
    for row in qrels:
        candidate_ids = row.get("candidate_ids") or []
        candidate_scores = row.get("candidate_scores") or []
        if len(candidate_ids) != len(candidate_scores):
            findings.append(
                _build_finding(
                    RELATION_CARDINALITY_MISMATCH,
                    dataset_name=dataset_name,
                    relation_id=str(row.get("relation_id")) if row.get("relation_id") else None,
                    message=(
                        "Qrels row has mismatched candidate_ids and candidate_scores lengths: "
                        f"{len(candidate_ids)} != {len(candidate_scores)}"
                    ),
                    metadata={
                        "candidate_count": len(candidate_ids),
                        "score_count": len(candidate_scores),
                    },
                )
            )
    return findings


def validate_retrieval_bundle(
    *,
    dataset_name: str,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> ValidationResult:
    """Run the shared validators against one retrieval artifact bundle."""

    findings = []
    findings.extend(validate_required_fields(queries, dataset_name=dataset_name, required_fields=("id",)))
    findings.extend(validate_required_fields(candidates, dataset_name=dataset_name, required_fields=("id",)))
    findings.extend(validate_id_uniqueness(queries, dataset_name=dataset_name, id_field="id"))
    findings.extend(validate_id_uniqueness(candidates, dataset_name=dataset_name, id_field="id"))
    findings.extend(_validate_prefixed_identifier(dataset_name=dataset_name, rows=queries, field_name="sample_id", expected_prefix="sample_"))
    findings.extend(_validate_prefixed_identifier(dataset_name=dataset_name, rows=candidates, field_name="sample_id", expected_prefix="sample_"))
    findings.extend(_validate_prefixed_identifier(dataset_name=dataset_name, rows=queries, field_name="asset_ids", expected_prefix="asset_"))
    findings.extend(_validate_prefixed_identifier(dataset_name=dataset_name, rows=candidates, field_name="asset_ids", expected_prefix="asset_"))
    findings.extend(
        _validate_prefixed_identifier(
            dataset_name=dataset_name,
            rows=qrels,
            field_name="relation_id",
            expected_prefix="relation_",
            relation_scope=True,
        )
    )
    findings.extend(_validate_relation_cardinality(dataset_name=dataset_name, qrels=qrels))
    findings.extend(validate_modality_constraints(queries, dataset_name=dataset_name))
    findings.extend(validate_modality_constraints(candidates, dataset_name=dataset_name))
    findings.extend(validate_media_payloads(queries, dataset_name=dataset_name))
    findings.extend(validate_media_payloads(candidates, dataset_name=dataset_name))
    findings.extend(
        validate_retrieval_integrity(
            dataset_name=dataset_name,
            queries=queries,
            candidates=candidates,
            qrels=qrels,
        )
    )
    return ValidationResult(dataset_name=dataset_name, findings=tuple(findings))
