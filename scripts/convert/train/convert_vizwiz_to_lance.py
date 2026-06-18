#!/usr/bin/env python
"""Convert VizWiz VQA training data into BToks Lance layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.vizwiz_train import ensure_vizwiz_indices, write_vizwiz_root


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing Annotations.zip and images zips.")
    parser.add_argument("--eval-root", type=Path, default=None, help="Optional MMEB-V2 VizWiz eval artifact root.")
    parser.add_argument("--output", type=Path, required=True, help="Output Lance dataset root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--split-batch-size", type=int, default=4096, help="Rows per split write batch.")
    parser.add_argument("--image-batch-size", type=int, default=512, help="Rows per image write batch.")
    parser.add_argument("--ensure-indices-only", action="store_true", help="Only create missing lookup indices.")
    return parser.parse_args()


def main() -> None:
    """Run conversion."""

    args = _parse_args()
    if args.ensure_indices_only:
        ensure_vizwiz_indices(args.output)
        print(json.dumps({"output_root": str(args.output), "indices": "ensured"}, ensure_ascii=False, indent=2))
        return
    summary = write_vizwiz_root(
        raw_root=args.raw_root,
        output_root=args.output,
        eval_root=args.eval_root,
        overwrite=args.overwrite,
        split_batch_size=args.split_batch_size,
        image_batch_size=args.image_batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
