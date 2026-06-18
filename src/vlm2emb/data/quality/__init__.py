"""Shared quality primitives for validation, cleaning, and reporting."""

from .cleaning import CleaningDecision, classify_validation_result
from .gates import preflight_retrieval_dataset
from .models import QualityFinding, ValidationResult
from .reason_codes import (
    BROKEN_MEDIA,
    EMPTY_QRELS,
    EMPTY_TEXT,
    ID_COLLISION,
    INVALID_IDENTIFIER,
    INVALID_POSITIVE,
    MISSING_CANDIDATE_LINK,
    MISSING_QUERY_LINK,
    MISSING_REFERENCE,
    RELATION_CARDINALITY_MISMATCH,
    REASON_CODE_REGISTRY,
    SCHEMA_MODALITY_CONSTRAINT_FAILED,
    SCHEMA_REQUIRED_FIELD_MISSING,
    get_reason_definition,
)

__all__ = [
    "BROKEN_MEDIA",
    "CleaningDecision",
    "EMPTY_QRELS",
    "EMPTY_TEXT",
    "ID_COLLISION",
    "INVALID_IDENTIFIER",
    "INVALID_POSITIVE",
    "MISSING_CANDIDATE_LINK",
    "MISSING_QUERY_LINK",
    "MISSING_REFERENCE",
    "RELATION_CARDINALITY_MISMATCH",
    "preflight_retrieval_dataset",
    "classify_validation_result",
    "QualityFinding",
    "REASON_CODE_REGISTRY",
    "SCHEMA_MODALITY_CONSTRAINT_FAILED",
    "SCHEMA_REQUIRED_FIELD_MISSING",
    "ValidationResult",
    "get_reason_definition",
]
