#!/usr/bin/env python
"""Generate deterministic cleaning decisions from serialized validation findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.quality.cleaning import (
    bucket_decisions,
    classify_validation_result,
    deserialize_validation_result,
    serialize_decision,
)
from vlm2emb.data.quality.reporting import write_quality_artifacts


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create keep/drop/quarantine outputs from validation findings")
    parser.add_argument("--findings", type=Path, required=True, help="Serialized validation result JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for cleaning outputs")
    args = parser.parse_args()

    with args.findings.open() as file_handle:
        payload = json.load(file_handle)
    result = deserialize_validation_result(payload)
    decisions = classify_validation_result(result)
    buckets = bucket_decisions(decisions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "quality.decisions.jsonl").open("w") as file_handle:
        for decision in decisions:
            file_handle.write(serialize_decision(decision) + "\n")

    _write_json(args.output_dir / "quality.keep.json", buckets["keep"])
    _write_json(args.output_dir / "quality.drop.json", buckets["drop"])
    _write_json(args.output_dir / "quality.quarantine.json", buckets["quarantine"])
    write_quality_artifacts(
        args.output_dir,
        result,
        decisions=decisions,
        artifact_locations=[
            {"kind": "decision_ledger", "path": str(args.output_dir / "quality.decisions.jsonl")},
            {"kind": "keep", "path": str(args.output_dir / "quality.keep.json")},
            {"kind": "drop", "path": str(args.output_dir / "quality.drop.json")},
            {"kind": "quarantine", "path": str(args.output_dir / "quality.quarantine.json")},
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
