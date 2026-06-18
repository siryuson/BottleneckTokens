#!/usr/bin/env python
"""CLI wrapper for YouCook2 training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.youcook2_train import ensure_youcook2_indices, write_youcook2_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, help="Local morpheushoc/youcook2 root or data directory")
    parser.add_argument(
        "--video-archive",
        type=Path,
        action="append",
        dest="video_archives",
        help="Raw video tar part; pass multiple times in part order",
    )
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--video-batch-size", type=int, default=16, help="Video rows per Lance write batch")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Annotation rows per Lance write batch")
    parser.add_argument(
        "--segment-max-frames",
        type=int,
        default=8,
        help="Maximum pre-extracted frames per row-level segment",
    )
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
    """Run the YouCook2 conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_youcook2_indices(args.output)
        return 0
    if args.parquet_root is None:
        raise ValueError("--parquet-root is required unless --ensure-indices-only is set.")
    if not args.video_archives:
        raise ValueError("--video-archive is required unless --ensure-indices-only is set.")
    summary = write_youcook2_root(
        parquet_root=args.parquet_root,
        video_archives=args.video_archives,
        output_root=args.output,
        overwrite=args.overwrite,
        video_batch_size=args.video_batch_size,
        split_batch_size=args.split_batch_size,
        segment_max_frames=args.segment_max_frames,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
