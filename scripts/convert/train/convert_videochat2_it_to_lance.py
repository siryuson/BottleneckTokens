#!/usr/bin/env python
"""CLI wrapper for VideoChat2-IT training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.videochat2_it_train import (
    ensure_videochat2_it_indices,
    write_videochat2_it_root,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, help="byminji/VideoChat2-IT-clean local root")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument(
        "--nextqa-frames",
        type=Path,
        default=Path("./data/NExTQA/data/frames.lance"),
        help="NExTQA frame-unit Lance table",
    )
    parser.add_argument(
        "--youcook2-frames",
        type=Path,
        default=Path("./data/YouCook2/data/frames.lance"),
        help="YouCook2 frame-unit Lance table",
    )
    parser.add_argument(
        "--ssv2-frames",
        type=Path,
        default=Path("./data/SmthSmthV2/data/frames.lance"),
        help="Something-Something-V2 frame-unit Lance table",
    )
    parser.add_argument(
        "--kinetics700-frames",
        type=Path,
        default=Path("./data/Kinetics-700/data/frames.lance"),
        help="Kinetics-700 frame-unit Lance table",
    )
    parser.add_argument(
        "--activitynet-videos",
        type=Path,
        default=Path("./data/ActivityNet-Captions/data/videos.lance"),
        help="ActivityNet full-video Lance table",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output/data if it exists")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for ActivityNet frame sampling")
    parser.add_argument("--frame-batch-size", type=int, default=16, help="ActivityNet frame rows per Lance batch")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Annotation rows per Lance write batch")
    parser.add_argument("--full-video-max-frames", type=int, default=64, help="Maximum sampled frames per full video")
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
    """Run the VideoChat2-IT conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_videochat2_it_indices(args.output)
        return 0
    if args.annotation_root is None:
        raise ValueError("--annotation-root is required unless --ensure-indices-only is set.")
    summary = write_videochat2_it_root(
        annotation_root=args.annotation_root,
        output_root=args.output,
        nextqa_frames=args.nextqa_frames,
        youcook2_frames=args.youcook2_frames,
        ssv2_frames=args.ssv2_frames,
        kinetics700_frames=args.kinetics700_frames,
        activitynet_videos=args.activitynet_videos,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        frame_batch_size=args.frame_batch_size,
        split_batch_size=args.split_batch_size,
        full_video_max_frames=args.full_video_max_frames,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
