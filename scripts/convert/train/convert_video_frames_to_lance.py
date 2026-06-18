#!/usr/bin/env python
"""Convert video frame directories into Lance format for training storage.

This tool belongs to the training-data conversion layer under scripts/convert/train.
It writes frame rows only and does not implement evaluation semantics.

This script converts a directory of video frames (organized as video_id/frame.jpeg)
to Lance format with schema: video_id, frame_idx, image.

Example usage:
    python scripts/convert/train/convert_video_frames_to_lance.py \
        --input ~/datasets/ShareGPTVideo/train_video_and_instruction/train_300k \
        --output /path/to/ShareGPTVideo_train_video_and_instruction \
        --temp-dir /tmp/lance_temp
"""

import argparse
import logging
import re
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


def natural_sort_key(filename: str) -> list[int | str]:
    """Natural sort key so frame_2 comes before frame_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", filename)]


class WriteProgress(FragmentWriteProgress):
    """Progress tracker for Lance write operations."""

    def __init__(self, estimated_fragments: int = 0):
        self.fragment_count = 0
        self.estimated_fragments = estimated_fragments

    def begin(self, fragment, **kwargs):
        self.fragment_count += 1
        if self.estimated_fragments > 0:
            logger.info(
                f"  Writing fragment {self.fragment_count}/{self.estimated_fragments}..."
            )
        else:
            logger.info(f"  Writing fragment {self.fragment_count}...")

    def complete(self, fragment, **kwargs):
        if self.estimated_fragments > 0:
            logger.info(
                f"  Fragment {self.fragment_count}/{self.estimated_fragments} complete"
            )
        else:
            logger.info(f"  Fragment {self.fragment_count} complete")


def iter_video_frames(
    input_dir: Path,
    batch_size: int = 1024,
) -> Iterator[pa.RecordBatch]:
    """Iterate over video frames and yield PyArrow RecordBatches.

    Args:
        input_dir: Directory containing video_id subdirectories.
        batch_size: Number of frames per batch.

    Yields:
        PyArrow RecordBatch with columns: video_id, frame_idx, image
    """
    schema = pa.schema([
        ("video_id", pa.string()),
        ("frame_idx", pa.string()),
        ("image", pa.binary()),
    ])

    video_ids = []
    frame_idxs = []
    images = []

    video_dirs = sorted(input_dir.iterdir())
    total_videos = len(video_dirs)

    for i, video_dir in enumerate(video_dirs):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name

        # Collect all frame files first, then sort globally to preserve temporal order
        frame_files = sorted(
            [
                *video_dir.glob("*.jpeg"),
                *video_dir.glob("*.jpg"),
            ],
            key=lambda path: natural_sort_key(path.name),
        )

        for frame_file in frame_files:
            video_ids.append(video_id)
            frame_idxs.append(frame_file.name)
            images.append(frame_file.read_bytes())

            # Yield batch when full
            if len(video_ids) >= batch_size:
                yield pa.RecordBatch.from_pydict(
                    {
                        "video_id": video_ids,
                        "frame_idx": frame_idxs,
                        "image": images,
                    },
                    schema=schema,
                )
                video_ids = []
                frame_idxs = []
                images = []

        # Log progress every 10000 videos
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{total_videos} videos...")

    # Yield remaining
    if video_ids:
        yield pa.RecordBatch.from_pydict(
            {
                "video_id": video_ids,
                "frame_idx": frame_idxs,
                "image": images,
            },
            schema=schema,
        )


def estimate_total_size(input_dir: Path, sample_count: int = 100) -> tuple[int, int]:
    """Estimate total size by sampling.

    Returns:
        (estimated_total_bytes, estimated_total_frames)
    """
    video_dirs = list(input_dir.iterdir())
    total_videos = len([d for d in video_dirs if d.is_dir()])

    # Sample some videos
    sample_bytes = 0
    sample_frames = 0
    sampled = 0

    for video_dir in video_dirs[:sample_count]:
        if not video_dir.is_dir():
            continue
        for f in video_dir.glob("*.jpeg"):
            sample_bytes += f.stat().st_size
            sample_frames += 1
        for f in video_dir.glob("*.jpg"):
            sample_bytes += f.stat().st_size
            sample_frames += 1
        sampled += 1

    if sampled == 0:
        return 0, 0

    avg_bytes_per_video = sample_bytes / sampled
    avg_frames_per_video = sample_frames / sampled

    estimated_total_bytes = int(avg_bytes_per_video * total_videos)
    estimated_total_frames = int(avg_frames_per_video * total_videos)

    return estimated_total_bytes, estimated_total_frames


def convert_video_frames(
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path | None = None,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    overwrite: bool = False,
) -> None:
    """Convert video frames to Lance format.

    Args:
        input_dir: Directory containing video_id subdirectories.
        output_dir: Output directory for Lance dataset.
        temp_dir: Optional local temp directory for remote filesystem compatibility.
        max_bytes_per_file: Maximum bytes per Lance file.
        overwrite: Whether to overwrite existing output.
    """
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Estimate size
    logger.info("Estimating dataset size...")
    estimated_bytes, estimated_frames = estimate_total_size(input_dir)
    estimated_fragments = max(1, estimated_bytes // max_bytes_per_file)

    logger.info(f"  Estimated total frames: {estimated_frames:,}")
    logger.info(f"  Estimated total size: {estimated_bytes / 1024**3:.2f} GB")
    logger.info(f"  Estimated fragments: {estimated_fragments}")

    # Setup output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Use input directory name as lance dataset name
    lance_name = f"{input_dir.name}.lance"
    final_output = data_dir / lance_name

    if final_output.exists() and not overwrite:
        logger.error(f"Output already exists: {final_output}. Use --overwrite to replace.")
        sys.exit(1)

    # Determine write path
    use_temp = temp_dir is not None
    if use_temp:
        temp_dir.mkdir(parents=True, exist_ok=True)
        write_path = temp_dir / lance_name
        if write_path.exists():
            shutil.rmtree(write_path)
        logger.info(f"Using temp directory: {temp_dir}")
    else:
        write_path = final_output
        if write_path.exists():
            shutil.rmtree(write_path)

    # Create RecordBatchReader from iterator
    schema = pa.schema([
        ("video_id", pa.string()),
        ("frame_idx", pa.string()),
        ("image", pa.binary()),
    ])

    reader = pa.RecordBatchReader.from_batches(
        schema,
        iter_video_frames(input_dir),
    )

    # Write to Lance
    logger.info("Writing to Lance format...")
    lance_dataset = lance.write_dataset(
        reader,
        str(write_path),
        mode="create",
        max_bytes_per_file=max_bytes_per_file,
        max_rows_per_group=1024,
        progress=WriteProgress(estimated_fragments),
    )

    row_count = lance_dataset.count_rows()
    logger.info(f"  Total rows written: {row_count:,}")

    # Copy to final destination if using temp
    if use_temp:
        logger.info(f"Copying to final destination: {final_output}")
        if final_output.exists():
            shutil.rmtree(final_output)
        shutil.copytree(write_path, final_output)
        shutil.rmtree(write_path)

    logger.info(f"Conversion complete: {final_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert video frames to Lance format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing video_id subdirectories",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for Lance dataset",
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

    convert_video_frames(
        input_dir=args.input,
        output_dir=args.output,
        temp_dir=args.temp_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
