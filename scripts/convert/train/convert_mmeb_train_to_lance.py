#!/usr/bin/env python
"""CLI wrapper for the MMEB-train source-preserving conversion.

The real conversion mainline now lives in
`src/vlm2emb/data/conversion/mmeb_train.py`. This script remains the stable
operator-facing entrypoint used by task plans, tests, and rerun commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm2emb.data.conversion.mmeb_train import (
    DEFAULT_IMAGE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    IMAGE_PATH_COLUMNS,
    IMAGE_SCHEMA,
    SAMPLE_SCHEMA,
    VARIANT_MAP,
    convert_images,
    convert_metadata,
    convert_mmeb_train_root,
    convert_mmeb_train_subset,
    discover_subsets,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the MMEB-train conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Raw MMEB-train root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output Lance root.",
    )
    parser.add_argument("--subset", action="append", help="Subset to convert. Can be repeated.")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Root for resolving raw image paths. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel image IO workers.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=DEFAULT_IMAGE_BATCH_SIZE,
        help="Number of image rows to read before writing one Lance batch.",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=LANCE_MAX_BYTES_PER_FILE,
        help="Lance max bytes per file. Defaults to 2 GiB.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the operator-facing conversion CLI."""

    args = build_parser().parse_args(argv)
    subsets = args.subset or discover_subsets(args.input_dir)
    for subset in subsets:
        summary = convert_mmeb_train_subset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            subset=subset,
            image_root=args.image_root,
            num_workers=args.num_workers,
            image_batch_size=args.image_batch_size,
            max_bytes_per_file=args.max_bytes_per_file,
        )
        split_desc = ", ".join(
            f"{split.name}:{split.rows}"
            + (f" dropped={split.dropped_rows}" if split.dropped_rows else "")
            for split in summary.splits
        )
        print(
            f"converted subset={summary.subset} splits=[{split_desc}] "
            f"images={summary.image_rows} image_table={summary.image_output_path}",
            flush=True,
        )
    return 0

__all__ = [
    "DEFAULT_IMAGE_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "IMAGE_PATH_COLUMNS",
    "IMAGE_SCHEMA",
    "LANCE_MAX_BYTES_PER_FILE",
    "SAMPLE_SCHEMA",
    "VARIANT_MAP",
    "build_parser",
    "convert_images",
    "convert_metadata",
    "convert_mmeb_train_root",
    "convert_mmeb_train_subset",
    "discover_subsets",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
