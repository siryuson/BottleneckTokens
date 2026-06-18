#!/usr/bin/env python
"""Download public Lance datasets from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        help="Hugging Face dataset repo id, for example $BTOKS_MMEB_V2_HF_REPO.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Directory where the Lance dataset tree will be downloaded.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision, branch, or commit.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional Hub allow pattern. Can be passed more than once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.local_dir),
        allow_patterns=args.allow_pattern,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
