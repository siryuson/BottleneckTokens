#!/usr/bin/env python
"""CLI wrapper for GQA training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.gqa_train import (
    ensure_gqa_train_indices,
    write_gqa_instruction_splits,
    write_gqa_train_root,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Local GQA root")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--eval-root", type=Path, default=None, help="Optional MMEB-V2 GQA eval root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--image-batch-size", type=int, default=1024, help="Image rows per Lance write batch")
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=LANCE_MAX_BYTES_PER_FILE,
        help="Lance max bytes per file; defaults to 2 GiB",
    )
    parser.add_argument(
        "--ensure-indices-only",
        action="store_true",
        help="Only add missing lookup indices to an existing converted root",
    )
    parser.add_argument(
        "--instruction-splits-only",
        action="store_true",
        help="Only write GQA instruction split tables to an existing converted root",
    )
    parser.add_argument(
        "--instruction-subset",
        choices=("all", "balanced"),
        action="append",
        default=None,
        help="Instruction subset to write when --instruction-splits-only is set; repeatable",
    )
    return parser.parse_args()


def main() -> int:
    """Run the GQA conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_gqa_train_indices(args.output)
        return 0
    if args.instruction_splits_only:
        summary = write_gqa_instruction_splits(
            raw_root=args.raw_root,
            output_root=args.output,
            eval_root=args.eval_root,
            overwrite=args.overwrite,
            instruction_subsets=tuple(args.instruction_subset or ("all", "balanced")),
            max_bytes_per_file=args.max_bytes_per_file,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    summary = write_gqa_train_root(
        raw_root=args.raw_root,
        output_root=args.output,
        eval_root=args.eval_root,
        overwrite=args.overwrite,
        image_batch_size=args.image_batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
