#!/usr/bin/env python
"""CLI wrapper for VATEX training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.vatex_train import ensure_vatex_indices, write_vatex_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, help="Official vatex_training_v1.0.json path")
    parser.add_argument("--video-archive", type=Path, help="Official vatex-dataset tar or tar.gz path")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--video-batch-size", type=int, default=16, help="Video rows per Lance write batch")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Annotation rows per Lance write batch")
    parser.add_argument("--full-video-max-frames", type=int, default=64, help="Maximum sampled frames per video")
    parser.add_argument(
        "--frame-unit-workers",
        type=int,
        default=1,
        help="Worker processes used to decode and JPEG-encode frame units",
    )
    parser.add_argument(
        "--skip-videos-table",
        action="store_true",
        help="Skip the raw video_bytes provenance table; training runtime only needs frames.lance",
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
    """Run the VATEX conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_vatex_indices(args.output)
        return 0
    if args.annotation is None:
        raise ValueError("--annotation is required unless --ensure-indices-only is set.")
    if args.video_archive is None:
        raise ValueError("--video-archive is required unless --ensure-indices-only is set.")
    summary = write_vatex_root(
        annotation_path=args.annotation,
        video_archive=args.video_archive,
        output_root=args.output,
        overwrite=args.overwrite,
        video_batch_size=args.video_batch_size,
        split_batch_size=args.split_batch_size,
        full_video_max_frames=args.full_video_max_frames,
        frame_unit_workers=args.frame_unit_workers,
        write_videos_table=not args.skip_videos_table,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
