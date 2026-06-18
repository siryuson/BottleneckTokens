"""Legacy BToks training entrypoint.

This module is kept as a compatibility shim. It mirrors ``scripts/train.py``
without shelling out to a repository-relative script path, so it still works
from an installed package and from non-repository working directories.
"""

from __future__ import annotations

import argparse
from logging import FileHandler, Formatter
import logging
import sys

from omegaconf import OmegaConf

from vlm2emb import train
from vlm2emb.config import ConfigLoader, apply_overrides, to_native_config
from vlm2emb.utils.logging import (
    DATE_FORMAT,
    LOG_FORMAT,
    add_rank_filter,
    set_verbosity,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse compatibility CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Deprecated BToks training shim; use scripts/train.py when running from the repository.",
    )
    parser.add_argument("config", help="Path to configuration file.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format, for example train.args.output_dir=./output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with function names and line numbers.",
    )
    parser.add_argument("--log-file", default=None, help="Optional path to a log file.")
    return parser.parse_args(argv)


def main() -> int:
    """Run training through the unified runtime path."""
    args = parse_args()
    print(
        "[DEPRECATED] `vlm2emb.cli.train_btoks` is deprecated; use the unified entrypoint:"
        " `python scripts/train.py <config.yaml> [overrides...]`",
        file=sys.stderr,
    )

    setup_logging("DEBUG" if args.verbose else "INFO")
    set_verbosity()
    if args.log_file:
        file_handler = FileHandler(args.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        add_rank_filter(file_handler)
        logging.getLogger().addHandler(file_handler)

    loader = ConfigLoader()
    config = loader.load_with_inheritance(str(args.config), resolve_interpolation=False)
    if args.overrides:
        config = apply_overrides(config, args.overrides)
    OmegaConf.resolve(config)
    runtime_config = to_native_config(config, resolve=True)
    train(runtime_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
