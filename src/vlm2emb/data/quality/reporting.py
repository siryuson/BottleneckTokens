"""Audit artifact builders for validation and cleaning outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cleaning import CleaningDecision
from .models import ValidationResult

QUALITY_AUDIT_NAME = "quality.audit.json"
QUALITY_REPORT_NAME = "quality.report.md"
QUALITY_VALIDATION_NAME = "quality.validation.json"


def build_quality_audit(
    result: ValidationResult,
    *,
    decisions: list[CleaningDecision] | None = None,
    artifact_locations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build one machine-readable quality audit payload."""

    decision_counts = {"keep": 0, "drop": 0, "quarantine": 0}
    for decision in decisions or []:
        decision_counts[decision.action] += 1

    ready = not result.has_blocking_errors and decision_counts["drop"] == 0 and decision_counts["quarantine"] == 0
    representative = [
        {
            "reason_code": finding.reason_code,
            "message": finding.message,
            "sample_id": finding.sample_id,
            "relation_id": finding.relation_id,
            "asset_id": finding.asset_id,
        }
        for finding in result.findings[:5]
    ]
    return {
        "dataset_name": result.dataset_name,
        "ready": ready,
        "summary": {
            "blocking_findings": result.has_blocking_errors,
            "finding_count": len(result.findings),
            "reason_counts": result.findings_by_reason,
            "decision_counts": decision_counts,
        },
        "representative_findings": representative,
        "artifact_locations": artifact_locations or [],
        "findings": [asdict(finding) for finding in result.findings],
        "decisions": [asdict(decision) for decision in decisions or []],
    }


def render_quality_report(audit: dict[str, Any]) -> str:
    """Render one human-readable Markdown quality report."""

    summary = audit["summary"]
    lines = [
        f"# Quality Report: {audit['dataset_name']}",
        "",
        f"- Ready: {'yes' if audit['ready'] else 'no'}",
        f"- Finding count: {summary['finding_count']}",
        f"- Blocking findings: {'yes' if summary['blocking_findings'] else 'no'}",
        f"- Keep / Drop / Quarantine: {summary['decision_counts']['keep']} / {summary['decision_counts']['drop']} / {summary['decision_counts']['quarantine']}",
        "",
        "## Reason Counts",
    ]
    if summary["reason_counts"]:
        for reason_code, count in sorted(summary["reason_counts"].items()):
            lines.append(f"- `{reason_code}`: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Representative Findings"])
    if audit["representative_findings"]:
        for finding in audit["representative_findings"]:
            lines.append(
                f"- `{finding['reason_code']}`: {finding['message']}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_quality_artifacts(
    output_dir: Path,
    result: ValidationResult,
    *,
    decisions: list[CleaningDecision] | None = None,
    artifact_locations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Write machine-readable and human-readable quality artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_quality_audit(
        result,
        decisions=decisions,
        artifact_locations=artifact_locations,
    )
    (output_dir / QUALITY_AUDIT_NAME).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    (output_dir / QUALITY_REPORT_NAME).write_text(render_quality_report(audit))
    (output_dir / QUALITY_VALIDATION_NAME).write_text(
        json.dumps(
            {
                "dataset_name": result.dataset_name,
                "metadata": result.metadata,
                "findings": [asdict(finding) for finding in result.findings],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return audit
