#!/usr/bin/env python
"""CLI wrapper for RefCOCO training Lance conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.refcoco_train import ensure_refcoco_indices, write_refcoco_root
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def parse_args() -> argparse.Namespace:
    """Parse conversion CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, help="Local jxu124/refcoco root")
    parser.add_argument("--coco-images-root", type=Path, help="Local COCO image root, zip root, tar root, or parquet root")
    parser.add_argument("--mmeb-v2-eval-root", type=Path, action="append", dest="mmeb_v2_eval_roots")
    parser.add_argument("--output", type=Path, required=True, help="Converted Lance root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--image-batch-size", type=int, default=1024, help="Image rows per Lance write batch")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Annotation rows per Lance write batch")
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
    """Run the RefCOCO conversion wrapper."""

    args = parse_args()
    if args.ensure_indices_only:
        ensure_refcoco_indices(args.output)
        return 0
    if args.parquet_root is None:
        raise ValueError("--parquet-root is required unless --ensure-indices-only is set.")
    if args.coco_images_root is None:
        raise ValueError("--coco-images-root is required unless --ensure-indices-only is set.")
    summary = write_refcoco_root(
        parquet_root=args.parquet_root,
        coco_images_root=args.coco_images_root,
        output_root=args.output,
        mmeb_v2_eval_roots=args.mmeb_v2_eval_roots,
        overwrite=args.overwrite,
        image_batch_size=args.image_batch_size,
        split_batch_size=args.split_batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
