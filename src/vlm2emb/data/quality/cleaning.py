"""Deterministic cleaning decisions derived from validation findings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .models import QualityFinding, ValidationResult

CleaningAction = Literal["keep", "drop", "quarantine"]
TargetType = Literal["dataset", "sample", "relation", "asset"]

ACTION_BY_DECISION_HINT: dict[str, CleaningAction] = {
    "keep": "keep",
    "drop": "drop",
    "quarantine": "quarantine",
    # Blocking findings must remain reviewable and must not silently downgrade to keep.
    "block": "quarantine",
}

ACTION_PRIORITY: dict[CleaningAction, int] = {
    "keep": 0,
    "drop": 1,
    "quarantine": 2,
}


@dataclass(frozen=True)
class CleaningDecision:
    """One deterministic cleaning decision for a single target entity."""

    dataset_name: str
    action: CleaningAction
    target_type: TargetType
    target_id: str
    reason_codes: tuple[str, ...]
    finding_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_target(finding: QualityFinding) -> tuple[TargetType, str]:
    if finding.asset_id:
        return "asset", finding.asset_id
    if finding.relation_id:
        return "relation", finding.relation_id
    if finding.sample_id:
        return "sample", finding.sample_id
    return "dataset", finding.dataset_name


def action_for_finding(finding: QualityFinding) -> CleaningAction:
    """Map one finding to its deterministic cleaning action."""

    return ACTION_BY_DECISION_HINT[finding.decision_hint]


def classify_validation_result(result: ValidationResult) -> list[CleaningDecision]:
    """Aggregate findings into deterministic record-level cleaning decisions."""

    grouped: dict[tuple[TargetType, str], list[QualityFinding]] = {}
    for finding in result.findings:
        target = _resolve_target(finding)
        grouped.setdefault(target, []).append(finding)

    decisions: list[CleaningDecision] = []
    for (target_type, target_id), findings in sorted(grouped.items(), key=lambda item: item[0]):
        actions = [action_for_finding(finding) for finding in findings]
        final_action = max(actions, key=lambda action: ACTION_PRIORITY[action])
        decisions.append(
            CleaningDecision(
                dataset_name=result.dataset_name,
                action=final_action,
                target_type=target_type,
                target_id=target_id,
                reason_codes=tuple(sorted({finding.reason_code for finding in findings})),
                finding_count=len(findings),
                metadata={"messages": [finding.message for finding in findings]},
            )
        )
    return decisions


def bucket_decisions(decisions: list[CleaningDecision]) -> dict[CleaningAction, dict[str, list[str]]]:
    """Group decisions by action and target type for CLI outputs."""

    buckets: dict[CleaningAction, dict[str, list[str]]] = {
        "keep": {"dataset": [], "sample": [], "relation": [], "asset": []},
        "drop": {"dataset": [], "sample": [], "relation": [], "asset": []},
        "quarantine": {"dataset": [], "sample": [], "relation": [], "asset": []},
    }
    for decision in decisions:
        buckets[decision.action][decision.target_type].append(decision.target_id)
    return buckets


def serialize_validation_result(result: ValidationResult) -> dict[str, Any]:
    """Serialize ValidationResult to a JSON-compatible payload."""

    return {
        "dataset_name": result.dataset_name,
        "metadata": result.metadata,
        "findings": [asdict(finding) for finding in result.findings],
    }


def deserialize_validation_result(payload: dict[str, Any]) -> ValidationResult:
    """Deserialize ValidationResult from a JSON-compatible payload."""

    findings = tuple(QualityFinding(**finding) for finding in payload.get("findings", []))
    return ValidationResult(
        dataset_name=payload["dataset_name"],
        findings=findings,
        metadata=payload.get("metadata", {}),
    )


def serialize_decision(decision: CleaningDecision) -> str:
    """Serialize one cleaning decision as a JSONL line."""

    return json.dumps(asdict(decision), ensure_ascii=False, sort_keys=True)
