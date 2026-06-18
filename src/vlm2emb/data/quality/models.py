"""Shared quality models for dataset validation and cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning"]
Scope = Literal["dataset", "sample", "relation", "asset"]
DecisionHint = Literal["block", "drop", "quarantine", "keep"]


@dataclass(frozen=True)
class QualityFinding:
    """One structured quality issue discovered during validation."""

    reason_code: str
    severity: Severity
    scope: Scope
    decision_hint: DecisionHint
    dataset_name: str
    message: str
    sample_id: str | None = None
    relation_id: str | None = None
    asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate typed findings for one dataset validation run."""

    dataset_name: str
    findings: tuple[QualityFinding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_blocking_errors(self) -> bool:
        """Return whether any finding should block ready status."""

        return any(
            finding.severity == "error" and finding.decision_hint in {"block", "drop", "quarantine"}
            for finding in self.findings
        )

    @property
    def findings_by_reason(self) -> dict[str, int]:
        """Count findings by stable reason code."""

        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.reason_code] = counts.get(finding.reason_code, 0) + 1
        return counts

    def extend(self, *groups: list[QualityFinding]) -> ValidationResult:
        """Return a new result with extra findings appended."""

        flattened = list(self.findings)
        for group in groups:
            flattened.extend(group)
        return ValidationResult(
            dataset_name=self.dataset_name,
            findings=tuple(flattened),
            metadata=dict(self.metadata),
        )
