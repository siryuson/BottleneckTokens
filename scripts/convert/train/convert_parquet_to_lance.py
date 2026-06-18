#!/usr/bin/env python
"""Convert a parquet dataset into Lance format for training storage.

This tool belongs to the training-data conversion layer under scripts/convert/train.
It preserves split structure and does not implement evaluation semantics.

This script converts HuggingFace-style Parquet datasets (with train/test splits)
to Lance format, preserving all fields and leveraging PyArrow's parallel I/O.

Example usage:
    python scripts/convert/train/convert_parquet_to_lance.py \
        --input /path/to/parquet_dataset \
        --output /path/to/lance_dataset

    # With custom splits
    python scripts/convert/train/convert_parquet_to_lance.py \
        --input /path/to/parquet_dataset \
        --output /path/to/lance_dataset \
        --splits train test val

    # Write via a local temp directory first, then copy
    python scripts/convert/train/convert_parquet_to_lance.py \
        --input /path/to/parquet_dataset \
        --output /path/to/lance_dataset \
        --temp-dir /tmp/lance_temp
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import lance
import pyarrow.dataset as ds
from lance.progress import FragmentWriteProgress

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WriteProgress(FragmentWriteProgress):
    """Progress tracker for Lance write operations."""

    def __init__(self, estimated_fragments: int = 0):
        self.fragment_count = 0
        self.estimated_fragments = estimated_fragments

    def begin(self, fragment, **kwargs):
        self.fragment_count += 1
        if self.estimated_fragments > 0:
            logger.info(
                f"    Writing fragment {self.fragment_count}/{self.estimated_fragments}..."
            )
        else:
            logger.info(f"    Writing fragment {self.fragment_count}...")

    def complete(self, fragment, **kwargs):
        if self.estimated_fragments > 0:
            logger.info(
                f"    Fragment {self.fragment_count}/{self.estimated_fragments} complete"
            )
        else:
            logger.info(f"    Fragment {self.fragment_count} complete")


def discover_splits(input_dir: Path) -> dict[str, list[Path]]:
    """Discover available splits from parquet files in the input directory.

    Parquet files are expected to follow the naming convention:
    {split}-{shard_idx}-of-{total_shards}.parquet

    Args:
        input_dir: Path to the input directory containing parquet files.

    Returns:
        Dictionary mapping split names to lists of parquet file paths.
    """
    data_dir = input_dir / "data"
    if not data_dir.exists():
        data_dir = input_dir

    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    splits: dict[str, list[Path]] = {}
    for pf in parquet_files:
        # Extract split name from filename (e.g., "train-00000-of-00082.parquet" -> "train")
        split_name = pf.stem.split("-")[0]
        if split_name not in splits:
            splits[split_name] = []
        splits[split_name].append(pf)

    # Sort files within each split
    for split_name in splits:
        splits[split_name].sort()

    return splits


def convert_split(
    parquet_files: list[Path],
    output_path: Path,
    mode: str = "create",
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> int:
    """Convert a single split from Parquet to Lance format.

    Uses PyArrow Dataset API for parallel reading of multiple parquet files.

    Args:
        parquet_files: List of parquet file paths for this split.
        output_path: Output path for the Lance dataset.
        mode: Write mode ('create' or 'overwrite').
        max_bytes_per_file: Maximum bytes per Lance file for auto-sharding.

    Returns:
        Number of rows written.
    """
    # Calculate total size for progress estimation
    total_bytes = sum(pf.stat().st_size for pf in parquet_files)
    estimated_fragments = max(1, (total_bytes + max_bytes_per_file - 1) // max_bytes_per_file)

    # Use PyArrow Dataset API for parallel reading
    # This automatically uses multiple threads for I/O
    dataset = ds.dataset(
        [str(pf) for pf in parquet_files],
        format="parquet",
    )

    logger.info(f"  Schema: {dataset.schema}")
    logger.info(f"  Files: {len(parquet_files)}, Total size: {total_bytes / 1024**3:.2f} GB")
    logger.info(f"  Estimated fragments: {estimated_fragments}")

    # Create a scanner with parallel reading enabled
    # PyArrow will automatically parallelize across files and row groups
    scanner = dataset.scanner(
        use_threads=True,
        batch_size=1024,  # Batch size for reading
    )

    # Write to Lance format
    # lance.write_dataset accepts Scanner directly and handles batching internally
    lance_dataset = lance.write_dataset(
        scanner,
        str(output_path),
        mode=mode,
        max_bytes_per_file=max_bytes_per_file,
        max_rows_per_group=1024,
        progress=WriteProgress(estimated_fragments),
    )

    return lance_dataset.count_rows()


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: list[str] | None = None,
    overwrite: bool = False,
    temp_dir: Path | None = None,
) -> None:
    """Convert a Parquet dataset to Lance format.

    Args:
        input_dir: Path to the input directory containing parquet files.
        output_dir: Path to the output directory for Lance datasets.
        splits: List of splits to convert. If None, all discovered splits are converted.
        overwrite: Whether to overwrite existing Lance datasets.
        temp_dir: Optional local temp directory for writing Lance files before copying
                  to the final destination. Useful for remote filesystems
                  that don't support atomic rename operations.
    """
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    if temp_dir:
        logger.info(f"Using temp directory: {temp_dir}")

    # Discover available splits
    discovered_splits = discover_splits(input_dir)
    logger.info(f"Discovered splits: {list(discovered_splits.keys())}")

    # Filter splits if specified
    if splits:
        splits_to_convert = {s: discovered_splits[s] for s in splits if s in discovered_splits}
        missing = set(splits) - set(splits_to_convert.keys())
        if missing:
            logger.warning(f"Requested splits not found: {missing}")
    else:
        splits_to_convert = discovered_splits

    if not splits_to_convert:
        raise ValueError("No splits to convert")

    # Create output directory: output_dir/data/
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine if we need to use temp directory
    use_temp = temp_dir is not None
    if use_temp:
        temp_dir.mkdir(parents=True, exist_ok=True)

    # Convert each split
    total_rows = 0
    for split_name, parquet_files in splits_to_convert.items():
        # Output as data/{split}.lance
        final_output = data_dir / f"{split_name}.lance"
        logger.info(f"Converting split '{split_name}' ({len(parquet_files)} files)...")

        if final_output.exists() and not overwrite:
            logger.warning(
                f"  Skipping '{split_name}': output already exists. Use --overwrite to replace."
            )
            continue

        # Determine write path (temp or final)
        if use_temp:
            write_path = temp_dir / f"{split_name}.lance"
            if write_path.exists():
                shutil.rmtree(write_path)
        else:
            write_path = final_output

        mode = "overwrite" if overwrite and write_path.exists() else "create"
        rows = convert_split(parquet_files, write_path, mode=mode)
        total_rows += rows
        logger.info(f"  Converted: {rows:,} rows")

        # Copy from temp to final destination if needed
        if use_temp:
            logger.info(f"  Copying to final destination: {final_output}")
            if final_output.exists():
                shutil.rmtree(final_output)
            shutil.copytree(write_path, final_output)
            shutil.rmtree(write_path)
            logger.info(f"  Completed: {final_output}")
        else:
            logger.info(f"  Completed: {write_path}")

    logger.info(f"Conversion complete. Total rows: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet dataset to Lance format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing parquet files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for Lance dataset",
    )
    parser.add_argument(
        "--splits", "-s",
        nargs="+",
        default=None,
        help="Splits to convert (default: all discovered splits)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Lance datasets",
    )
    parser.add_argument(
        "--temp-dir", "-t",
        type=Path,
        default=None,
        help="Local temp directory for writing Lance files before copying to final destination. "
             "Required for remote filesystems that don't support atomic rename.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        splits=args.splits,
        overwrite=args.overwrite,
        temp_dir=args.temp_dir,
    )


if __name__ == "__main__":
    main()
