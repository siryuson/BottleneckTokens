#!/usr/bin/env python
"""Run shared dataset validation against one standardized retrieval dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.quality.gates import (
    REQUIRED_RETRIEVAL_ARTIFACTS,
    validate_document_train_dataset,
    validate_retrieval_dataset,
)
from vlm2emb.data.quality.reporting import write_quality_artifacts


def _has_retrieval_artifacts(dataset_path: Path) -> bool:
    return all((dataset_path / artifact_name).exists() for artifact_name in REQUIRED_RETRIEVAL_ARTIFACTS)


def _has_train_lance(dataset_path: Path) -> bool:
    return (dataset_path / "data" / "train.lance").exists() or (dataset_path / "train.lance").exists()


def _load_manifest(dataset_path: Path) -> dict:
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as file_handle:
        return json.load(file_handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one standardized dataset")
    parser.add_argument("--path", type=Path, required=True, help="Dataset directory containing Lance artifacts")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for audit outputs (default: dataset path)")
    parser.add_argument(
        "--media-sample-limit",
        type=int,
        default=64,
        help="Number of train-lance media rows to decode for train-only document datasets",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    args = parser.parse_args()

    try:
        artifact_locations: list[dict[str, str]]
        if _has_retrieval_artifacts(args.path):
            metadata, result = validate_retrieval_dataset(args.path)
            artifact_locations = [
                {"kind": "queries", "path": str(args.path / "queries.lance")},
                {"kind": "candidates", "path": str(args.path / "candidates.lance")},
                {"kind": "qrels", "path": str(args.path / "qrels.lance")},
                {"kind": "metadata", "path": str(args.path / "metadata.json")},
                {"kind": "manifest", "path": str(args.path / "manifest.json")},
            ]
        else:
            manifest = _load_manifest(args.path)
            if manifest.get("task_type") == "document_retrieval" and _has_train_lance(args.path):
                metadata, result = validate_document_train_dataset(
                    args.path,
                    media_sample_limit=args.media_sample_limit,
                )
                train_lance_path = args.path / "data" / "train.lance"
                if not train_lance_path.exists():
                    train_lance_path = args.path / "train.lance"
                artifact_locations = [
                    {"kind": "train_lance", "path": str(train_lance_path)},
                    {"kind": "manifest", "path": str(args.path / "manifest.json")},
                ]
            else:
                raise ValueError(
                    "Unsupported dataset layout for validate_dataset.py. "
                    "Expected either retrieval artifacts or a document_retrieval train.lance root."
                )
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Validation failed: {exc}")
        return 1

    output_dir = args.output_dir or args.path
    audit = write_quality_artifacts(
        output_dir,
        result,
        artifact_locations=artifact_locations,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "status": "passed" if audit["ready"] else "failed",
                    "dataset": metadata.get("name"),
                    "audit_path": str((output_dir / "quality.audit.json")),
                },
                ensure_ascii=False,
            )
        )
    else:
        print(f"Validation {'passed' if audit['ready'] else 'failed'}: {metadata.get('name', args.path.name)}")
    return 0 if audit["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
