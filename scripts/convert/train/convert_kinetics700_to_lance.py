#!/usr/bin/env python
"""CLI wrapper for Kinetics-700 training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.kinetics700_train import (
    ensure_kinetics700_indices,
    write_kinetics700_root,
    write_kinetics700_root_sharded,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, help="Raw Kinetics-700-2020 root with annotations/ and train_tars/")
    parser.add_argument("--vlm2vec-root", type=Path, help="Local VLM2Vec/Kinetics-700 metadata root")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel tar readers for video IO/probing")
    parser.add_argument(
        "--shard-workers",
        type=int,
        default=1,
        help="Process-level tar shard workers; values > 1 use the sharded converter.",
    )
    parser.add_argument(
        "--threads-per-shard",
        type=int,
        default=1,
        help="Reader threads inside each process-level shard worker.",
    )
    parser.add_argument("--video-batch-size", type=int, default=64, help="Video rows per Lance write batch")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Metadata rows per Lance write batch")
    parser.add_argument("--merge-batch-size", type=int, default=128, help="Rows per batch when merging shard tables")
    parser.add_argument("--full-video-max-frames", type=int, default=64, help="Maximum sampled frames per video")
    parser.add_argument(
        "--skip-invalid-videos",
        action="store_true",
        help="Drop videos that cannot be decoded and record them under data/exclusions/invalid_videos.lance",
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
    parser.add_argument("--keep-shards", action="store_true", help="Keep temporary shard roots after final merge")
    return parser.parse_args()


def main() -> int:
    """Run the Kinetics-700 conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_kinetics700_indices(args.output)
        return 0
    if args.raw_root is None or args.vlm2vec_root is None:
        raise ValueError("--raw-root and --vlm2vec-root are required unless --ensure-indices-only is set.")
    common_kwargs = {
        "raw_root": args.raw_root,
        "vlm2vec_root": args.vlm2vec_root,
        "output_root": args.output,
        "overwrite": args.overwrite,
        "video_batch_size": args.video_batch_size,
        "split_batch_size": args.split_batch_size,
        "max_bytes_per_file": args.max_bytes_per_file,
        "invalid_video_policy": "skip" if args.skip_invalid_videos else "error",
        "full_video_max_frames": args.full_video_max_frames,
    }
    if args.shard_workers > 1:
        summary = write_kinetics700_root_sharded(
            **common_kwargs,
            shard_workers=args.shard_workers,
            threads_per_shard=args.threads_per_shard,
            merge_batch_size=args.merge_batch_size,
            keep_shards=args.keep_shards,
        )
    else:
        summary = write_kinetics700_root(
            **common_kwargs,
            num_workers=args.num_workers,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
