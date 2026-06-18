"""Shared media validators that reuse runtime decode semantics."""

from __future__ import annotations

from typing import Any

from vlm2emb.data.datasets.const import decode_image, extract_image_bytes

from ..models import QualityFinding
from ..reason_codes import BROKEN_MEDIA, EMPTY_TEXT, get_reason_definition


def _build_finding(
    reason_code: str,
    *,
    dataset_name: str,
    message: str,
    sample_id: str | None = None,
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
        asset_id=asset_id,
        metadata=metadata or {},
    )


def validate_text_not_empty(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    text_field: str = "text",
    sample_id_field: str = "sample_id",
) -> list[QualityFinding]:
    """Return findings for rows whose required text payload is empty."""

    findings: list[QualityFinding] = []
    for row in rows:
        text = row.get(text_field, "")
        if isinstance(text, str) and text.strip():
            continue
        findings.append(_build_finding(
            EMPTY_TEXT,
            dataset_name=dataset_name,
            message=f"Empty text payload in field {text_field}",
            sample_id=str(row.get(sample_id_field)) if row.get(sample_id_field) else None,
            metadata={"field": text_field},
        ))
    return findings


def validate_media_payloads(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    media_field: str = "images",
    single_media_field: str = "image",
    sample_id_field: str = "sample_id",
    asset_ids_field: str = "asset_ids",
) -> list[QualityFinding]:
    """Return findings for media bytes that cannot be decoded."""

    findings: list[QualityFinding] = []
    for row in rows:
        sample_id = str(row.get(sample_id_field)) if row.get(sample_id_field) else None
        asset_ids = row.get(asset_ids_field) or []
        media_items = row.get(media_field) or []
        if not media_items and row.get(single_media_field) is not None:
            media_items = [row.get(single_media_field)]
        for index, media in enumerate(media_items):
            media_bytes = extract_image_bytes(media)
            if media_bytes is None:
                continue
            try:
                decode_image(media_bytes)
            except Exception as exc:  # pragma: no cover - decode errors depend on PIL internals
                asset_id = str(asset_ids[index]) if index < len(asset_ids) else None
                findings.append(_build_finding(
                    BROKEN_MEDIA,
                    dataset_name=dataset_name,
                    message=f"Unreadable media payload at index {index}: {exc}",
                    sample_id=sample_id,
                    asset_id=asset_id,
                    metadata={"field": media_field, "index": index},
                ))
    return findings
