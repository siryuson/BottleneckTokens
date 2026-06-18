"""Preflight gates that fail fast before training or evaluation starts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import lance

from vlm2emb.data.datasets.const import decode_image, extract_image_bytes

from .models import QualityFinding, ValidationResult
from .reporting import QUALITY_AUDIT_NAME
from .reason_codes import BROKEN_MEDIA, EMPTY_TEXT, SCHEMA_REQUIRED_FIELD_MISSING, get_reason_definition
from .validators.runtime import validate_retrieval_bundle

REQUIRED_RETRIEVAL_ARTIFACTS = (
    "queries.lance",
    "candidates.lance",
    "qrels.lance",
    "metadata.json",
)

PreflightMode = Literal["auto", "light", "full", "off"]

TRAIN_DOCUMENT_REQUIRED_TEXT_FIELDS_BY_LOADER: dict[str, tuple[str, ...]] = {
    "vidore_train": ("query", "answer", "source"),
    "vidore_rag_train": ("query", "answer", "source"),
    "visrag_train": ("query", "source"),
}


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


def _load_lance_rows(path: Path, *, batch_size: int = 256) -> list[dict[str, Any]]:
    """Load one Lance table incrementally to avoid giant single-table materialization."""

    dataset = lance.dataset(str(path))
    rows: list[dict[str, Any]] = []
    scanner = dataset.scanner(batch_size=batch_size)
    for batch in scanner.to_batches():
        rows.extend(batch.to_pylist())
    return rows


def _load_retrieval_bundle(dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with (dataset_path / "metadata.json").open() as file_handle:
        metadata = json.load(file_handle)
    queries = _load_lance_rows(dataset_path / "queries.lance", batch_size=64)
    candidates = _load_lance_rows(dataset_path / "candidates.lance", batch_size=64)
    qrels = _load_lance_rows(dataset_path / "qrels.lance", batch_size=512)
    return metadata, queries, candidates, qrels


def validate_retrieval_dataset(path: str | Path):
    """Run direct retrieval validation without applying the audit shortcut."""

    dataset_path = Path(path)
    missing = [name for name in REQUIRED_RETRIEVAL_ARTIFACTS if not (dataset_path / name).exists()]
    if missing:
        raise ValueError(f"Missing required retrieval artifacts: {missing}")

    metadata, queries, candidates, qrels = _load_retrieval_bundle(dataset_path)
    dataset_name = str(metadata.get("name") or dataset_path.name)
    result = validate_retrieval_bundle(
        dataset_name=dataset_name,
        queries=queries,
        candidates=candidates,
        qrels=qrels,
    )
    return metadata, result


def _resolve_train_lance_path(dataset_path: Path) -> Path:
    candidates = (
        dataset_path / "data" / "train.lance",
        dataset_path / "train.lance",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(
        f"Could not find training Lance artifact under {dataset_path}. "
        "Expected data/train.lance or train.lance."
    )


def _load_manifest(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing manifest.json under {dataset_path}")
    with manifest_path.open() as file_handle:
        return json.load(file_handle)


def validate_document_train_dataset(
    path: str | Path,
    *,
    media_sample_limit: int = 64,
) -> tuple[dict[str, Any], ValidationResult]:
    """Run quality validation for standardized train-only document retrieval roots."""

    dataset_path = Path(path)
    manifest = _load_manifest(dataset_path)
    dataset_name = str(manifest.get("dataset_name") or dataset_path.name)
    metadata = manifest.get("metadata", {}) if isinstance(manifest.get("metadata", {}), dict) else {}
    loader_type = str(metadata.get("loader_type", "") or "")
    required_text_fields = TRAIN_DOCUMENT_REQUIRED_TEXT_FIELDS_BY_LOADER.get(loader_type)
    if required_text_fields is None:
        raise ValueError(f"Unsupported document train loader_type for validation: {loader_type or '<missing>'}")

    lance_path = _resolve_train_lance_path(dataset_path)
    dataset = lance.dataset(str(lance_path))
    findings: list[QualityFinding] = []
    row_count = 0

    scanner = dataset.scanner(columns=list(required_text_fields), batch_size=512, batch_readahead=2)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            row_count += 1
            sample_id = f"sample_row_{row_count}"
            for field_name in required_text_fields:
                value = row.get(field_name)
                if value is None:
                    findings.append(
                        _build_finding(
                            SCHEMA_REQUIRED_FIELD_MISSING,
                            dataset_name=dataset_name,
                            message=f"Missing required field: {field_name}",
                            sample_id=sample_id,
                            metadata={"field": field_name},
                        )
                    )
                    continue
                if not isinstance(value, str) or not value.strip():
                    findings.append(
                        _build_finding(
                            EMPTY_TEXT,
                            dataset_name=dataset_name,
                            message=f"Empty text payload in field {field_name}",
                            sample_id=sample_id,
                            metadata={"field": field_name},
                        )
                    )

    if row_count == 0:
        findings.append(
            _build_finding(
                SCHEMA_REQUIRED_FIELD_MISSING,
                dataset_name=dataset_name,
                message="Training Lance dataset contains no rows.",
                metadata={"field": "train_rows"},
            )
        )

    image_checks = 0
    image_scanner = dataset.scanner(columns=["image"], batch_size=min(max(media_sample_limit, 1), 64))
    for batch in image_scanner.to_batches():
        for row in batch.to_pylist():
            if image_checks >= media_sample_limit:
                break
            image_checks += 1
            sample_id = f"sample_row_{image_checks}"
            image_bytes = extract_image_bytes(row.get("image"))
            if image_bytes is None:
                findings.append(
                    _build_finding(
                        SCHEMA_REQUIRED_FIELD_MISSING,
                        dataset_name=dataset_name,
                        message="Missing required field: image",
                        sample_id=sample_id,
                        metadata={"field": "image"},
                    )
                )
                continue
            try:
                decode_image(image_bytes)
            except Exception as exc:  # pragma: no cover - decode errors depend on PIL internals
                findings.append(
                    _build_finding(
                        BROKEN_MEDIA,
                        dataset_name=dataset_name,
                        message=f"Unreadable media payload in sampled train row: {exc}",
                        sample_id=sample_id,
                        asset_id=f"asset_train_row_{image_checks}",
                        metadata={"field": "image", "sample_index": image_checks},
                    )
                )
        if image_checks >= media_sample_limit:
            break

    return (
        {"name": dataset_name, "task_type": manifest.get("task_type", ""), "loader_type": loader_type},
        ValidationResult(
            dataset_name=dataset_name,
            findings=tuple(findings),
            metadata={
                "task_type": manifest.get("task_type", ""),
                "loader_type": loader_type,
                "row_count": row_count,
                "media_sample_limit": media_sample_limit,
                "media_samples_checked": image_checks,
                "validation_mode": "document_train_lance_sampled_media",
            },
        ),
    )


def _check_required_artifacts(dataset_path: Path) -> None:
    missing = [name for name in REQUIRED_RETRIEVAL_ARTIFACTS if not (dataset_path / name).exists()]
    if missing:
        raise ValueError(f"Missing required retrieval artifacts: {missing}")


def summarize_validation_result(result) -> str:
    """Build a compact fail-fast message from a ValidationResult."""

    if not result.findings:
        return "dataset is ready"
    pieces = []
    for reason_code, count in sorted(result.findings_by_reason.items()):
        pieces.append(f"{reason_code}={count}")
    return ", ".join(pieces)


def preflight_retrieval_dataset(
    path: str | Path,
    *,
    mode: PreflightMode = "auto",
) -> dict[str, Any]:
    """Validate one retrieval dataset directory before evaluator construction."""

    dataset_path = Path(path)
    audit_path = dataset_path / QUALITY_AUDIT_NAME

    if mode not in {"auto", "light", "full", "off"}:
        raise ValueError(f"Unsupported preflight mode: {mode}")

    if mode == "off":
        metadata_path = dataset_path / "metadata.json"
        with metadata_path.open() as file_handle:
            return json.load(file_handle)

    _check_required_artifacts(dataset_path)

    if audit_path.exists():
        with audit_path.open() as file_handle:
            audit = json.load(file_handle)
        if not audit.get("ready", False):
            reason_counts = audit.get("summary", {}).get("reason_counts", {})
            pieces = [f"{code}={count}" for code, count in sorted(reason_counts.items())]
            raise ValueError(
                f"Dataset preflight failed for {audit.get('dataset_name', dataset_path.name)}: "
                f"{', '.join(pieces) or 'audit marks dataset not ready'}"
            )
        metadata_path = dataset_path / "metadata.json"
        with metadata_path.open() as file_handle:
            return json.load(file_handle)

    if mode == "light":
        metadata_path = dataset_path / "metadata.json"
        with metadata_path.open() as file_handle:
            return json.load(file_handle)

    if mode == "auto":
        mode = "full"

    metadata, result = validate_retrieval_dataset(dataset_path)
    dataset_name = str(metadata.get("name") or dataset_path.name)
    if result.has_blocking_errors:
        raise ValueError(f"Dataset preflight failed for {dataset_name}: {summarize_validation_result(result)}")
    return metadata
