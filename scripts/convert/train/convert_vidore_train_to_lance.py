#!/usr/bin/env python
"""CLI wrapper for raw-preserving ViDoRe training conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm2emb.data.conversion.vidore_train import (
    VIDORE_RAG_SUBSET_SOURCES,
    convert_vidore_root,
)
from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for ViDoRe training conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Raw ViDoRe dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for the full view.")
    parser.add_argument(
        "--subset-output-base",
        type=Path,
        help="Optional base directory for source-specific vidore_rag_train roots.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="ViDoRe source-specific subset to write. Can be repeated. Defaults to all known sources.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Arrow scan batch size.")
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=LANCE_MAX_BYTES_PER_FILE,
        help="Lance max bytes per file. Defaults to 2 GiB.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def _resolve_sources(requested: list[str] | None) -> tuple[str, ...]:
    """Resolve and validate ViDoRe source names."""

    if not requested:
        return VIDORE_RAG_SUBSET_SOURCES
    unknown = sorted(set(requested) - set(VIDORE_RAG_SUBSET_SOURCES))
    if unknown:
        raise ValueError(f"Unknown ViDoRe source(s): {unknown}")
    return tuple(requested)


def main(argv: list[str] | None = None) -> int:
    """Run the operator-facing ViDoRe conversion CLI."""

    args = build_parser().parse_args(argv)
    sources = _resolve_sources(args.source)
    summary = convert_vidore_root(
        args.input_dir,
        args.output_dir,
        rag_output_base=args.subset_output_base,
        rag_sources=sources,
        batch_size=args.batch_size,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    payload = {
        "family": "vidore",
        "full": {
            "output_path": summary.full.output_path,
            "rows": summary.full.rows,
        },
        "rag_subsets": [
            {"output_path": item.output_path, "rows": item.rows}
            for item in summary.rag_subsets
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"full rows={summary.full.rows} output={summary.full.output_path}")
        for item in summary.rag_subsets:
            print(f"rag subset rows={item.rows} output={item.output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
