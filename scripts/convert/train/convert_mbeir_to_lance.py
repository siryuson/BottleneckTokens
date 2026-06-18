#!/usr/bin/env python
"""CLI wrapper for converting M-BEIR train tasks to Lance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.mbeir_train import MBEIR_TASK_CONFIGS, write_mbeir_train_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the M-BEIR converter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to the raw M-BEIR root.")
    parser.add_argument("--output", type=Path, required=True, help="Output Lance root.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(MBEIR_TASK_CONFIGS),
        help="Dataset task to convert. Repeat to select multiple tasks. Defaults to all supported tasks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output root if it already exists.")
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--candidate-batch-size", type=int, default=8192)
    parser.add_argument("--image-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for small-file image reads.")
    parser.add_argument("--max-bytes-per-file", type=int, default=LANCE_MAX_BYTES_PER_FILE)
    return parser


def main() -> None:
    """Run the M-BEIR conversion."""

    args = build_parser().parse_args()
    summary = write_mbeir_train_root(
        raw_root=args.raw_root,
        output_root=args.output,
        dataset_names=args.datasets,
        overwrite=args.overwrite,
        query_batch_size=args.query_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        image_batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
