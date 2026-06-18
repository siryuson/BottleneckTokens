#!/usr/bin/env python
"""Convert JSONL files into Lance format for training storage.

This tool belongs to the training-data conversion layer under scripts/convert/train.
It preserves directory structure and does not implement evaluation semantics.

This script converts JSONL files to Lance format, preserving directory structure.
Each .jsonl file becomes a .lance dataset.

Example usage:
    python scripts/convert/train/convert_jsonl_to_lance.py \
        --input ~/datasets/video_instruction \
        --output /path/to/video_instruction \
        --temp-dir /tmp/lance_temp
"""

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import lance
import pyarrow as pa
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
                f"      Writing fragment {self.fragment_count}/{self.estimated_fragments}..."
            )
        else:
            logger.info(f"      Writing fragment {self.fragment_count}...")

    def complete(self, fragment, **kwargs):
        pass


def infer_schema_from_jsonl(jsonl_path: Path, sample_lines: int = 100) -> pa.Schema:
    """Infer PyArrow schema from JSONL file by sampling lines."""
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"Empty JSONL file: {jsonl_path}")

    # Convert samples to PyArrow table to infer schema
    table = pa.Table.from_pylist(samples)
    return table.schema


def iter_jsonl_batches(
    jsonl_path: Path,
    schema: pa.Schema,
    batch_size: int = 1024,
) -> Iterator[pa.RecordBatch]:
    """Iterate over JSONL file and yield PyArrow RecordBatches."""
    rows = []
    line_count = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            line_count += 1

            if len(rows) >= batch_size:
                table = pa.Table.from_pylist(rows, schema=schema)
                yield table.to_batches()[0]
                rows = []

            if line_count % 100000 == 0:
                logger.info(f"      Processed {line_count:,} lines...")

    # Yield remaining
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
        yield table.to_batches()[0]


def count_lines(jsonl_path: Path) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(jsonl_path, "rb") as f:
        for _ in f:
            count += 1
    return count


def convert_jsonl_file(
    jsonl_path: Path,
    output_path: Path,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
) -> int:
    """Convert a single JSONL file to Lance format.

    Returns:
        Number of rows written.
    """
    # Count lines and estimate size
    file_size = jsonl_path.stat().st_size
    line_count = count_lines(jsonl_path)
    estimated_fragments = max(1, file_size // max_bytes_per_file)

    logger.info(f"    Lines: {line_count:,}, Size: {file_size / 1024**2:.1f} MB")

    # Infer schema
    schema = infer_schema_from_jsonl(jsonl_path)
    logger.info(f"    Schema: {schema.names}")

    # Create reader
    reader = pa.RecordBatchReader.from_batches(
        schema,
        iter_jsonl_batches(jsonl_path, schema),
    )

    # Write to Lance
    lance_dataset = lance.write_dataset(
        reader,
        str(output_path),
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
        max_rows_per_group=1024,
        progress=WriteProgress(estimated_fragments),
    )

    return lance_dataset.count_rows()


def discover_jsonl_files(input_dir: Path) -> list[Path]:
    """Discover all JSONL files in directory recursively."""
    return sorted(input_dir.rglob("*.jsonl"))


def convert_jsonl_directory(
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Convert all JSONL files in a directory to Lance format.

    Preserves directory structure.
    """
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Discover all JSONL files
    jsonl_files = discover_jsonl_files(input_dir)
    logger.info(f"Found {len(jsonl_files)} JSONL files")

    if not jsonl_files:
        logger.warning("No JSONL files found")
        return

    use_temp = temp_dir is not None
    if use_temp:
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temp directory: {temp_dir}")

    total_rows = 0
    for i, jsonl_path in enumerate(jsonl_files, 1):
        # Compute relative path to preserve directory structure
        rel_path = jsonl_path.relative_to(input_dir)
        lance_name = rel_path.with_suffix(".lance")

        final_output = output_dir / lance_name
        logger.info(f"[{i}/{len(jsonl_files)}] Converting {rel_path}...")

        if final_output.exists() and not overwrite:
            logger.warning("  Skipping: output already exists. Use --overwrite to replace.")
            continue

        # Create parent directories
        final_output.parent.mkdir(parents=True, exist_ok=True)

        # Determine write path
        if use_temp:
            write_path = temp_dir / lance_name.name
            if write_path.exists():
                shutil.rmtree(write_path)
        else:
            write_path = final_output
            if write_path.exists():
                shutil.rmtree(write_path)

        # Convert
        rows = convert_jsonl_file(jsonl_path, write_path)
        total_rows += rows
        logger.info(f"    Converted: {rows:,} rows")

        # Copy to final destination if using temp
        if use_temp:
            logger.info(f"    Copying to: {final_output}")
            if final_output.exists():
                shutil.rmtree(final_output)
            shutil.copytree(write_path, final_output)
            shutil.rmtree(write_path)

    logger.info(f"Conversion complete. Total rows: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL files to Lance format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for Lance datasets",
    )
    parser.add_argument(
        "--temp-dir", "-t",
        type=Path,
        default=None,
        help="Local temp directory for remote filesystem compatibility",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
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

    convert_jsonl_directory(
        input_dir=args.input,
        output_dir=args.output,
        temp_dir=args.temp_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
