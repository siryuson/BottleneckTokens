"""Run MMEB-V2 evaluation conversion from explicit dataset selections.

This module owns the user-facing CLI for MMEB-V2 conversion work. It parses
dataset selection arguments, adapts shared CLI inputs into pipeline-specific
function signatures, and dispatches each dataset directly to one concrete
conversion function.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

from vlm2emb.data.conversion.mmeb_v2.datasets import (
    get_dataset_names,
    get_modality,
    get_pipeline,
)
from vlm2emb.data.conversion.mmeb_v2.image import (
    convert_image_cls_dataset,
    convert_image_i2i_vg_dataset,
    convert_image_i2t_dataset,
    convert_image_qa_dataset,
    convert_image_t2i_dataset,
)
from vlm2emb.data.conversion.mmeb_v2.video import (
    convert_video_cls_dataset,
    convert_video_momentseeker_dataset,
    convert_video_mret_dataset,
    convert_video_qa_dataset,
    convert_video_ret_dataset,
)
from vlm2emb.data.conversion.mmeb_v2.visdoc import convert_visdoc_dataset

PipelineAdapter: TypeAlias = Callable[
    [str, Path],
    Path,
]


def _build_pipeline_adapters() -> dict[str, Callable[..., Path]]:
    """Build one explicit CLI-to-pipeline parameter adapter table.

    This layer exists because concrete pipeline functions do not yet share one
    unified signature. The CLI accepts one normalized parameter set:

    - ``dataset_name``
    - ``output_root``
    - ``mmeb_v2_root``
    - ``mmeb_eval_root``
    - ``source_overrides``
    - ``max_workers``

    Each adapter forwards only the subset actually consumed by one pipeline and
    renames parameters where current implementations still differ.

    This is an explicit parameter adaptation layer, not a second dispatch layer.
    The actual dispatch key is already fixed by ``dataset -> pipeline``.
    """
    return {
        "image_cls": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_image_cls_dataset(
            dataset_name,
            output_root,
            mmeb_eval_root=mmeb_eval_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "image_qa": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_image_qa_dataset(
            dataset_name,
            output_root,
            mmeb_eval_root=mmeb_eval_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "image_i2t": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_image_i2t_dataset(
            dataset_name,
            output_root,
            mmeb_eval_root=mmeb_eval_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "image_t2i": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_image_t2i_dataset(
            dataset_name,
            output_root,
            mmeb_eval_root=mmeb_eval_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "image_i2i_vg": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_image_i2i_vg_dataset(
            dataset_name,
            output_root,
            mmeb_eval_root=mmeb_eval_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "video_cls": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_video_cls_dataset(
            dataset_name,
            output_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "video_qa": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_video_qa_dataset(
            dataset_name,
            output_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "video_text_retrieval": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_video_ret_dataset(
            dataset_name,
            output_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "video_moment_retrieval": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_video_mret_dataset(
            dataset_name,
            output_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "video_momentseeker": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_video_momentseeker_dataset(
            dataset_name,
            output_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
            max_workers=max_workers,
        ),
        "visdoc_beir": lambda dataset_name, output_root, *, mmeb_v2_root, mmeb_eval_root, source_overrides, max_workers: convert_visdoc_dataset(
            dataset_name,
            output_root,
            mmeb_v2_root=mmeb_v2_root,
            source_overrides=source_overrides,
        ),
    }


def parse_source_overrides(entries: list[str] | None) -> dict[str, Path]:
    """Parse repeatable ``KEY=PATH`` overrides from the CLI."""

    overrides: dict[str, Path] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --source-path value {entry!r}; expected KEY=PATH"
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid --source-path value {entry!r}; expected KEY=PATH"
            )
        overrides[key] = Path(value).expanduser()
    return overrides


def parse_datasets_arg(value: str) -> list[str]:
    """Parse one comma-separated ``--datasets`` argument."""

    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset name")
    unknown = sorted(set(datasets) - set(get_dataset_names()))
    if unknown:
        raise ValueError(f"Unknown MMEB-V2 datasets: {unknown}")

    deduped: list[str] = []
    seen: set[str] = set()
    for dataset_name in datasets:
        if dataset_name in seen:
            continue
        seen.add(dataset_name)
        deduped.append(dataset_name)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    """Build the MMEB-V2 conversion CLI parser."""

    parser = argparse.ArgumentParser(
        description="Convert MMEB-V2 evaluation datasets into Lance artifacts.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--datasets",
        help="Comma-separated MMEB-V2 dataset names to convert.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Convert all registered MMEB-V2 datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root directory for converted artifacts.",
    )
    parser.add_argument(
        "--mmeb-v2-root",
        type=Path,
        help="Local extracted root for MMEB-V2.",
    )
    parser.add_argument(
        "--mmeb-eval-root",
        type=Path,
        help="Local extracted root for MMEB image annotations (MMEB-eval/MMEB_Test_Instruct).",
    )
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        help="Explicit source override in KEY=PATH form. May be repeated.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Optional max worker count for parallel media loading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved conversion plan without writing artifacts.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all registered MMEB-V2 datasets and exit.",
    )
    return parser


def _resolve_selected_datasets(args: argparse.Namespace) -> list[str]:
    """Resolve the final dataset list from ``--datasets`` or ``--all``."""
    if args.all:
        return list(get_dataset_names())
    if args.datasets:
        return parse_datasets_arg(args.datasets)
    raise ValueError("One of --datasets or --all is required unless --list-datasets is used")


def _print_dataset_names(dataset_names: Iterable[str]) -> None:
    """Print dataset names line by line for ``--list-datasets`` output."""
    for dataset_name in dataset_names:
        print(dataset_name)


def convert_selected_datasets(
    dataset_names: list[str],
    *,
    output_root: Path,
    mmeb_v2_root: Path | None = None,
    mmeb_eval_root: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
    max_workers: int = 8,
    dry_run: bool = False,
) -> list[Path]:
    """Convert one explicit list of MMEB-V2 datasets."""

    outputs: list[Path] = []
    pipeline_adapters = _build_pipeline_adapters()
    for dataset_name in dataset_names:
        pipeline = get_pipeline(dataset_name)
        modality = get_modality(dataset_name)
        if dry_run:
            print(f"{dataset_name}\t{modality}\t{pipeline}")
            continue
        invoke = pipeline_adapters.get(pipeline)
        if invoke is None:
            raise ValueError(
                f"Unsupported MMEB-V2 pipeline for {dataset_name}: {pipeline} "
                f"(modality={modality})"
            )
        outputs.append(
            invoke(
                dataset_name,
                output_root,
                mmeb_v2_root=mmeb_v2_root,
                mmeb_eval_root=mmeb_eval_root,
                source_overrides=source_overrides,
                max_workers=max_workers,
            )
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    """Run the MMEB-V2 conversion CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        _print_dataset_names(get_dataset_names())
        return 0

    if args.output_root is None:
        parser.error("--output-root is required unless --list-datasets is used")

    if not args.all and not args.datasets:
        parser.error("one of --datasets or --all is required unless --list-datasets is used")

    dataset_names = _resolve_selected_datasets(args)
    source_overrides = parse_source_overrides(args.source_path)
    convert_selected_datasets(
        dataset_names,
        output_root=args.output_root,
        mmeb_v2_root=args.mmeb_v2_root,
        mmeb_eval_root=args.mmeb_eval_root,
        source_overrides=source_overrides,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    return 0


__all__ = [
    "build_parser",
    "convert_selected_datasets",
    "main",
    "parse_datasets_arg",
    "parse_source_overrides",
]


if __name__ == "__main__":
    raise SystemExit(main())
