"""Shared retrieval integrity validators."""

from __future__ import annotations

from typing import Any

from ..models import QualityFinding
from ..reason_codes import (
    EMPTY_QRELS,
    INVALID_POSITIVE,
    MISSING_CANDIDATE_LINK,
    MISSING_QUERY_LINK,
    get_reason_definition,
)


def _build_finding(
    reason_code: str,
    *,
    dataset_name: str,
    message: str,
    relation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> QualityFinding:
    definition = get_reason_definition(reason_code)
    return QualityFinding(
        reason_code=reason_code,
        severity=definition.severity,
        scope=definition.scope,
        decision_hint=definition.decision_hint,
        dataset_name=dataset_name,
        message=message,
        relation_id=relation_id,
        metadata=metadata or {},
    )


def validate_retrieval_integrity(
    *,
    dataset_name: str,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> list[QualityFinding]:
    """Return findings for retrieval-specific integrity failures."""

    if not qrels:
        return [
            _build_finding(
                EMPTY_QRELS,
                dataset_name=dataset_name,
                message="Retrieval dataset has no qrels rows.",
            )
        ]

    findings: list[QualityFinding] = []
    query_ids = {str(row["id"]) for row in queries if row.get("id") is not None}
    candidate_ids = {str(row["id"]) for row in candidates if row.get("id") is not None}

    for row in qrels:
        relation_id = str(row.get("relation_id")) if row.get("relation_id") else None
        query_id = str(row.get("query_id", ""))
        if query_id not in query_ids:
            findings.append(_build_finding(
                MISSING_QUERY_LINK,
                dataset_name=dataset_name,
                relation_id=relation_id,
                message=f"Qrels references missing query: {query_id}",
                metadata={"query_id": query_id},
            ))

        row_candidate_ids = [str(value) for value in (row.get("candidate_ids") or [])]
        row_scores = [float(value) for value in (row.get("candidate_scores") or [])]
        if not row_candidate_ids:
            findings.append(_build_finding(
                EMPTY_QRELS,
                dataset_name=dataset_name,
                relation_id=relation_id,
                message="Qrels row contains no candidate ids.",
            ))
            continue

        missing_candidates = [candidate_id for candidate_id in row_candidate_ids if candidate_id not in candidate_ids]
        for candidate_id in missing_candidates:
            findings.append(_build_finding(
                MISSING_CANDIDATE_LINK,
                dataset_name=dataset_name,
                relation_id=relation_id,
                message=f"Qrels references missing candidate: {candidate_id}",
                metadata={"candidate_id": candidate_id},
            ))

        if not any(score > 0 for score in row_scores):
            findings.append(_build_finding(
                INVALID_POSITIVE,
                dataset_name=dataset_name,
                relation_id=relation_id,
                message="Qrels row contains no positive candidate score.",
            ))

    return findings
