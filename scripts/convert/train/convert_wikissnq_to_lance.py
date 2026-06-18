#!/usr/bin/env python
"""CLI wrapper for converting Wiki-SS-NQ train data to Lance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.wikissnq_train import write_wikissnq_train_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the Wiki-SS-NQ converter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-root", type=Path, required=True, help="Path to Tevatron/wiki-ss-nq.")
    parser.add_argument("--corpus-root", type=Path, required=True, help="Path to Tevatron/wiki-ss-corpus-new.")
    parser.add_argument("--output", type=Path, required=True, help="Output Lance root.")
    parser.add_argument("--eval-root", type=Path, default=None, help="Optional MMEB-V2 Wiki-SS-NQ eval Lance root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output root if it already exists.")
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--image-batch-size", type=int, default=1024)
    parser.add_argument("--max-bytes-per-file", type=int, default=LANCE_MAX_BYTES_PER_FILE)
    return parser


def main() -> None:
    """Run the Wiki-SS-NQ conversion."""

    args = build_parser().parse_args()
    summary = write_wikissnq_train_root(
        query_root=args.query_root,
        corpus_root=args.corpus_root,
        output_root=args.output,
        eval_root=args.eval_root,
        overwrite=args.overwrite,
        sample_batch_size=args.sample_batch_size,
        image_batch_size=args.image_batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
