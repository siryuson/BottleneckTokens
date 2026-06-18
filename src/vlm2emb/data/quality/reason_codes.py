"""Stable reason codes shared by validation, cleaning, and reporting."""

from __future__ import annotations

from dataclasses import dataclass

from .models import DecisionHint, Scope, Severity


@dataclass(frozen=True)
class ReasonCodeDefinition:
    """One stable quality reason code definition."""

    code: str
    severity: Severity
    scope: Scope
    decision_hint: DecisionHint
    description: str


SCHEMA_REQUIRED_FIELD_MISSING = "schema_required_field_missing"
SCHEMA_MODALITY_CONSTRAINT_FAILED = "schema_modality_constraint_failed"
EMPTY_TEXT = "empty_text"
BROKEN_MEDIA = "broken_media"
MISSING_REFERENCE = "missing_reference"
ID_COLLISION = "id_collision"
EMPTY_QRELS = "empty_qrels"
MISSING_QUERY_LINK = "missing_query_link"
MISSING_CANDIDATE_LINK = "missing_candidate_link"
INVALID_POSITIVE = "invalid_positive"
INVALID_IDENTIFIER = "invalid_identifier"
RELATION_CARDINALITY_MISMATCH = "relation_cardinality_mismatch"

REASON_CODE_REGISTRY: dict[str, ReasonCodeDefinition] = {
    SCHEMA_REQUIRED_FIELD_MISSING: ReasonCodeDefinition(
        code=SCHEMA_REQUIRED_FIELD_MISSING,
        severity="error",
        scope="sample",
        decision_hint="block",
        description="A required field is missing or empty.",
    ),
    SCHEMA_MODALITY_CONSTRAINT_FAILED: ReasonCodeDefinition(
        code=SCHEMA_MODALITY_CONSTRAINT_FAILED,
        severity="error",
        scope="sample",
        decision_hint="block",
        description="A modality-specific schema constraint failed.",
    ),
    EMPTY_TEXT: ReasonCodeDefinition(
        code=EMPTY_TEXT,
        severity="error",
        scope="sample",
        decision_hint="drop",
        description="A required text payload is empty.",
    ),
    BROKEN_MEDIA: ReasonCodeDefinition(
        code=BROKEN_MEDIA,
        severity="error",
        scope="asset",
        decision_hint="quarantine",
        description="Media bytes cannot be decoded or read reliably.",
    ),
    MISSING_REFERENCE: ReasonCodeDefinition(
        code=MISSING_REFERENCE,
        severity="error",
        scope="sample",
        decision_hint="block",
        description="A required linked record or asset is missing.",
    ),
    ID_COLLISION: ReasonCodeDefinition(
        code=ID_COLLISION,
        severity="error",
        scope="sample",
        decision_hint="block",
        description="A supposedly stable identifier is duplicated.",
    ),
    EMPTY_QRELS: ReasonCodeDefinition(
        code=EMPTY_QRELS,
        severity="error",
        scope="relation",
        decision_hint="block",
        description="A retrieval dataset has no qrels or no positive judgments.",
    ),
    MISSING_QUERY_LINK: ReasonCodeDefinition(
        code=MISSING_QUERY_LINK,
        severity="error",
        scope="relation",
        decision_hint="block",
        description="A qrels row points to a missing query.",
    ),
    MISSING_CANDIDATE_LINK: ReasonCodeDefinition(
        code=MISSING_CANDIDATE_LINK,
        severity="error",
        scope="relation",
        decision_hint="block",
        description="A qrels row points to a missing candidate.",
    ),
    INVALID_POSITIVE: ReasonCodeDefinition(
        code=INVALID_POSITIVE,
        severity="error",
        scope="relation",
        decision_hint="block",
        description="A retrieval relation has no positive candidate.",
    ),
    INVALID_IDENTIFIER: ReasonCodeDefinition(
        code=INVALID_IDENTIFIER,
        severity="error",
        scope="sample",
        decision_hint="block",
        description="A supposedly stable identifier does not match the expected canonical prefix.",
    ),
    RELATION_CARDINALITY_MISMATCH: ReasonCodeDefinition(
        code=RELATION_CARDINALITY_MISMATCH,
        severity="error",
        scope="relation",
        decision_hint="block",
        description="A retrieval relation has mismatched candidate and score cardinalities.",
    ),
}


def get_reason_definition(code: str) -> ReasonCodeDefinition:
    """Return the registered definition for one reason code."""

    return REASON_CODE_REGISTRY[code]
