#!/usr/bin/env python
"""CLI wrapper for raw-preserving LLaVA-Hound training conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm2emb.data.conversion.llavahound_train import (
    convert_llavahound_root,
    discover_instruction_subsets,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for LLaVA-Hound training conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Source LLaVA-Hound root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output LLaVA-Hound root.")
    parser.add_argument(
        "--subset",
        action="append",
        help="Instruction subset below data/video_instruction without .lance. Can be repeated.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Lance scan batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel file-copy workers for existing Lance tables.",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=LANCE_MAX_BYTES_PER_FILE,
        help="Lance max bytes per file. Defaults to 2 GiB.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def convert_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run conversion from parsed CLI arguments and return a JSON-ready summary."""

    subsets = tuple(args.subset) if args.subset else discover_instruction_subsets(args.input_dir)
    summary = convert_llavahound_root(
        args.input_dir,
        args.output_dir,
        instruction_subsets=subsets,
        batch_size=args.batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
        num_workers=args.num_workers,
    )
    return {
        "family": "llavahound",
        "root": str(args.output_dir),
        "frames": {
            "output_path": summary.frames.output_path,
            "rows": summary.frames.rows,
        },
        "instructions": [
            {"output_path": item.output_path, "rows": item.rows}
            for item in summary.instructions
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the operator-facing LLaVA-Hound conversion CLI."""

    args = build_parser().parse_args(argv)
    payload = convert_from_args(args)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"frames rows={payload['frames']['rows']} output={payload['frames']['output_path']}")
        for item in payload["instructions"]:
            print(f"instruction rows={item['rows']} output={item['output_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
