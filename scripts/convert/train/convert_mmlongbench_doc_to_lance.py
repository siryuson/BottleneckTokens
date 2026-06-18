#!/usr/bin/env python
"""CLI wrapper for converting MMLongBench-Doc train data to Lance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.mmlongbench_doc_train import (
    ensure_mmlongbench_doc_train_indices,
    write_mmlongbench_doc_train_root,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the MMLongBench-Doc converter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local yubo2333/MMLongBench-Doc root.")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="MMEB-V2 MMLongBench-doc eval root used to exclude leaked train rows.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Rows per Lance write batch.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes for PDF page rendering.")
    parser.add_argument("--max-bytes-per-file", type=int, default=LANCE_MAX_BYTES_PER_FILE)
    parser.add_argument(
        "--ensure-indices-only",
        action="store_true",
        help="Only add missing lookup indices to an existing converted root.",
    )
    return parser


def main() -> int:
    """Run MMLongBench-Doc conversion."""

    args = build_parser().parse_args()
    if args.ensure_indices_only:
        ensure_mmlongbench_doc_train_indices(args.output)
        return 0
    summary = write_mmlongbench_doc_train_root(
        input_root=args.input,
        output_root=args.output,
        eval_root=args.eval_root,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
