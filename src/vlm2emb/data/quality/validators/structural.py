"""Shared structural validators for standardized dataset artifacts."""

from __future__ import annotations

from typing import Any

from ..models import QualityFinding
from ..reason_codes import (
    ID_COLLISION,
    MISSING_REFERENCE,
    SCHEMA_MODALITY_CONSTRAINT_FAILED,
    SCHEMA_REQUIRED_FIELD_MISSING,
    get_reason_definition,
)


def _build_finding(
    reason_code: str,
    *,
    dataset_name: str,
    message: str,
    sample_id: str | None = None,
    relation_id: str | None = None,
    asset_id: str | None = None,
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
        sample_id=sample_id,
        relation_id=relation_id,
        asset_id=asset_id,
        metadata=metadata or {},
    )


def validate_required_fields(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    required_fields: tuple[str, ...],
    sample_id_field: str = "sample_id",
) -> list[QualityFinding]:
    """Return findings for rows that are missing required fields."""

    findings: list[QualityFinding] = []
    for row in rows:
        sample_id = row.get(sample_id_field)
        for field_name in required_fields:
            value = row.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                findings.append(_build_finding(
                    SCHEMA_REQUIRED_FIELD_MISSING,
                    dataset_name=dataset_name,
                    message=f"Missing required field: {field_name}",
                    sample_id=str(sample_id) if sample_id else None,
                    metadata={"field": field_name},
                ))
    return findings


def validate_id_uniqueness(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    id_field: str,
    sample_id_field: str = "sample_id",
) -> list[QualityFinding]:
    """Return findings for duplicated stable identifiers."""

    seen: dict[str, str | None] = {}
    findings: list[QualityFinding] = []
    for row in rows:
        value = row.get(id_field)
        if value in (None, ""):
            continue
        value = str(value)
        sample_id = str(row.get(sample_id_field)) if row.get(sample_id_field) else None
        if value in seen:
            findings.append(_build_finding(
                ID_COLLISION,
                dataset_name=dataset_name,
                message=f"Duplicate identifier detected in {id_field}: {value}",
                sample_id=sample_id,
                metadata={"field": id_field, "value": value},
            ))
        else:
            seen[value] = sample_id
    return findings


def validate_modality_constraints(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    text_field: str = "text",
    images_field: str = "images",
    image_field: str = "image",
) -> list[QualityFinding]:
    """Return findings for rows that violate basic modality expectations."""

    findings: list[QualityFinding] = []
    for row in rows:
        sample_id = str(row.get("sample_id")) if row.get("sample_id") else None
        has_text = bool((row.get(text_field) or "").strip()) if isinstance(row.get(text_field), str) else bool(row.get(text_field))
        has_images = bool(row.get(images_field) or [])
        if not has_images and row.get(image_field) is not None:
            has_images = True
        if not has_text and not has_images:
            findings.append(_build_finding(
                SCHEMA_MODALITY_CONSTRAINT_FAILED,
                dataset_name=dataset_name,
                message="Row has neither usable text nor media payload.",
                sample_id=sample_id,
            ))
    return findings


def validate_reference_links(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    field_name: str,
    existing_ids: set[str],
    relation_id_field: str = "relation_id",
) -> list[QualityFinding]:
    """Return findings for rows that point to missing linked identifiers."""

    findings: list[QualityFinding] = []
    for row in rows:
        relation_id = str(row.get(relation_id_field)) if row.get(relation_id_field) else None
        values = row.get(field_name)
        if not values:
            continue
        if not isinstance(values, list):
            values = [values]
        missing = [str(value) for value in values if str(value) not in existing_ids]
        for missing_id in missing:
            findings.append(_build_finding(
                MISSING_REFERENCE,
                dataset_name=dataset_name,
                message=f"Missing linked reference in {field_name}: {missing_id}",
                relation_id=relation_id,
                metadata={"field": field_name, "missing_id": missing_id},
            ))
    return findings
