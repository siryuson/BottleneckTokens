"""LLaVA-Hound raw-preserving Lance conversion helpers.

The current LLaVA-Hound source in this repository is already normalized into a
dual-Lance layout: one shared frame table and multiple instruction tables. This
module keeps the conversion boundary narrow by copying those raw tables into a
chosen output root without creating manifests, dataset infos, README files, or
preformatted training views.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import lance

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE

LLAVAHOUND_FULL_ROOT_NAME = "ShareGPTVideo_train_video_and_instruction"
LLAVAHOUND_FRAME_TABLE = "train_300k"
LLAVAHOUND_INSTRUCTION_PREFIX = Path("data") / "video_instruction"


@dataclass(frozen=True)
class LanceWriteSummary:
    """Summary for one converted Lance table.

    Attributes:
        output_path: Destination Lance table path.
        rows: Number of copied rows.
    """

    output_path: str
    rows: int


@dataclass(frozen=True)
class LlavaHoundConversionSummary:
    """Summary for a LLaVA-Hound dual-table conversion.

    Attributes:
        frames: Copied shared frame table summary.
        instructions: Copied instruction table summaries, one per subset.
    """

    frames: LanceWriteSummary
    instructions: tuple[LanceWriteSummary, ...]


def discover_instruction_subsets(input_dir: str | Path) -> tuple[str, ...]:
    """Discover instruction table subsets under ``data/video_instruction``.

    Args:
        input_dir: Source LLaVA-Hound root.

    Returns:
        Relative subset names without the ``.lance`` suffix, for example
        ``("train/sft/video_caption_300k",)``.

    Raises:
        FileNotFoundError: If the instruction root does not exist or contains no
            Lance tables.
    """

    instruction_root = Path(input_dir) / LLAVAHOUND_INSTRUCTION_PREFIX
    if not instruction_root.is_dir():
        raise FileNotFoundError(f"LLaVA-Hound instruction root not found: {instruction_root}")
    subset_names: list[str] = []
    for path in sorted(instruction_root.rglob("*.lance")):
        relative = path.relative_to(instruction_root)
        if any(part.endswith(".lance") for part in relative.parts[:-1]):
            continue
        subset_names.append(str(relative).removesuffix(".lance"))
    subsets = tuple(subset_names)
    if not subsets:
        raise FileNotFoundError(f"No LLaVA-Hound instruction Lance tables found under {instruction_root}")
    return subsets


def _copy_lance_table(
    input_path: Path,
    output_path: Path,
    *,
    num_workers: int,
) -> int:
    """Copy one existing Lance table directory and return its row count.

    The source LLaVA-Hound data is already a raw-preserving Lance layout. A
    directory-level copy is both faster and more faithful than scanning binary
    frame rows and rewriting them, because it keeps table metadata and existing
    scalar indexes intact.
    """

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError(f"LLaVA-Hound input and output Lance table paths must differ: {input_path}")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Lance table not found: {input_path}")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive for LLaVA-Hound Lance table copy.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{uuid4().hex}")
    backup_path = output_path.with_name(f".{output_path.name}.backup-{uuid4().hex}")
    lock_path = output_path.with_name(f".{output_path.name}.copy.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        file_pairs: list[tuple[Path, Path]] = []
        for source in input_path.rglob("*"):
            relative = source.relative_to(input_path)
            target = temp_path / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                file_pairs.append((source, target))

        def copy_one(pair: tuple[Path, Path]) -> None:
            source, target = pair
            shutil.copy2(source, target)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(copy_one, file_pairs))

        rows = lance.dataset(str(temp_path)).count_rows()
        if output_path.exists():
            output_path.rename(backup_path)
        temp_path.rename(output_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)
        return rows
    except Exception:
        if backup_path.exists() and not output_path.exists():
            backup_path.rename(output_path)
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def convert_llavahound_frame_table(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    num_workers: int = 8,
) -> LanceWriteSummary:
    """Copy the shared frame table into the output root.

    Args:
        input_dir: Source LLaVA-Hound root.
        output_dir: Destination LLaVA-Hound root.
        batch_size: Retained for API symmetry; directory-copy conversion does
            not scan Lance batches.
        max_bytes_per_file: Retained for API symmetry; existing Lance files are
            copied as-is.
        num_workers: Number of parallel file-copy workers.

    Returns:
        Summary for the copied frame table.
    """

    del batch_size, max_bytes_per_file
    input_path = Path(input_dir) / "data" / f"{LLAVAHOUND_FRAME_TABLE}.lance"
    output_path = Path(output_dir) / "data" / f"{LLAVAHOUND_FRAME_TABLE}.lance"
    rows = _copy_lance_table(
        input_path,
        output_path,
        num_workers=num_workers,
    )
    return LanceWriteSummary(output_path=str(output_path), rows=rows)


def convert_llavahound_instruction_table(
    input_dir: str | Path,
    output_dir: str | Path,
    subset: str,
    *,
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    num_workers: int = 8,
) -> LanceWriteSummary:
    """Copy one raw instruction table into the output root.

    Args:
        input_dir: Source LLaVA-Hound root.
        output_dir: Destination LLaVA-Hound root.
        subset: Relative path below ``data/video_instruction`` without the
            ``.lance`` suffix.
        batch_size: Retained for API symmetry; directory-copy conversion does
            not scan Lance batches.
        max_bytes_per_file: Retained for API symmetry; existing Lance files are
            copied as-is.
        num_workers: Number of parallel file-copy workers.

    Returns:
        Summary for the copied instruction table.
    """

    del batch_size, max_bytes_per_file
    input_path = Path(input_dir) / LLAVAHOUND_INSTRUCTION_PREFIX / f"{subset}.lance"
    output_path = Path(output_dir) / LLAVAHOUND_INSTRUCTION_PREFIX / f"{subset}.lance"
    rows = _copy_lance_table(
        input_path,
        output_path,
        num_workers=num_workers,
    )
    return LanceWriteSummary(output_path=str(output_path), rows=rows)


def convert_llavahound_root(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    instruction_subsets: tuple[str, ...] | None = None,
    batch_size: int = 512,
    max_bytes_per_file: int = LANCE_MAX_BYTES_PER_FILE,
    num_workers: int = 8,
) -> LlavaHoundConversionSummary:
    """Convert a LLaVA-Hound root into the raw-preserving rerun layout.

    Args:
        input_dir: Source root containing ``data/train_300k.lance`` and
            ``data/video_instruction/**/*.lance``.
        output_dir: Destination root.
        instruction_subsets: Optional subset list. When omitted, all discovered
            instruction Lance tables are copied.
        batch_size: Retained for API symmetry; directory-copy conversion does
            not scan Lance batches.
        max_bytes_per_file: Retained for API symmetry; existing Lance files are
            copied as-is.
        num_workers: Number of parallel file-copy workers.

    Returns:
        Conversion summary for the frame table and instruction tables.
    """

    subsets = instruction_subsets or discover_instruction_subsets(input_dir)
    frames = convert_llavahound_frame_table(
        input_dir,
        output_dir,
        batch_size=batch_size,
        max_bytes_per_file=max_bytes_per_file,
        num_workers=num_workers,
    )
    instructions = tuple(
        convert_llavahound_instruction_table(
            input_dir,
            output_dir,
            subset,
            batch_size=batch_size,
            max_bytes_per_file=max_bytes_per_file,
            num_workers=num_workers,
        )
        for subset in subsets
    )
    return LlavaHoundConversionSummary(frames=frames, instructions=instructions)


__all__ = [
    "LANCE_MAX_BYTES_PER_FILE",
    "LLAVAHOUND_FRAME_TABLE",
    "LLAVAHOUND_FULL_ROOT_NAME",
    "LLAVAHOUND_INSTRUCTION_PREFIX",
    "LanceWriteSummary",
    "LlavaHoundConversionSummary",
    "convert_llavahound_frame_table",
    "convert_llavahound_instruction_table",
    "convert_llavahound_root",
    "discover_instruction_subsets",
]
