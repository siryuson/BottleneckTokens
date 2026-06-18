#!/usr/bin/env python
"""CLI wrapper for HMDB51 training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.hmdb51_train import ensure_hmdb51_indices, write_hmdb51_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, help="HMDB51 root with test_train_splits/")
    parser.add_argument("--vlm2vec-root", type=Path, help="Local VLM2Vec/HMDB51 metadata root")
    parser.add_argument("--video-root", type=Path, help="Extracted HMDB51 video root organized as class_name/*.avi")
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
    """Run the HMDB51 conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_hmdb51_indices(args.output)
        return 0
    if args.raw_root is None or args.vlm2vec_root is None or args.video_root is None:
        raise ValueError(
            "--raw-root, --vlm2vec-root and --video-root are required unless --ensure-indices-only is set."
        )
    summary = write_hmdb51_root(
        raw_root=args.raw_root,
        vlm2vec_root=args.vlm2vec_root,
        video_root=args.video_root,
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
