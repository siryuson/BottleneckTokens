#!/usr/bin/env python
"""CLI wrapper for Place365 training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.place365_train import (
    ensure_place365_indices,
    write_place365_root,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Local Place365 root")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for image rows")
    parser.add_argument("--image-batch-size", type=int, default=2048, help="Image rows per Lance write batch")
    parser.add_argument("--split-batch-size", type=int, default=8192, help="Metadata rows per Lance write batch")
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
    return parser.parse_args()


def main() -> int:
    """Run the Place365 conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_place365_indices(args.output)
        return 0
    summary = write_place365_root(
        raw_root=args.raw_root,
        output_root=args.output,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        image_batch_size=args.image_batch_size,
        split_batch_size=args.split_batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
