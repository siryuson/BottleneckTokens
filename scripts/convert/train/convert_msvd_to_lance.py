#!/usr/bin/env python
"""CLI wrapper for MSVD training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.msvd_train import ensure_msvd_indices, write_msvd_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Local MSVD root with split JSON files and raw_videos/")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for video IO/probing")
    parser.add_argument("--video-batch-size", type=int, default=64, help="Video rows per Lance write batch")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Metadata rows per Lance write batch")
    parser.add_argument("--full-video-max-frames", type=int, default=64, help="Maximum sampled frames per video")
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
    """Run the MSVD conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_msvd_indices(args.output)
        return 0
    if args.source_root is None:
        raise ValueError("--source-root is required unless --ensure-indices-only is set.")
    summary = write_msvd_root(
        source_root=args.source_root,
        output_root=args.output,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        video_batch_size=args.video_batch_size,
        split_batch_size=args.split_batch_size,
        full_video_max_frames=args.full_video_max_frames,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
