#!/usr/bin/env python3
"""Training entry point for BToks and VLM2Vec retrieval models.

Usage:
    # Via launcher (recommended)
    python scripts/train.py configs/presets/vlm2vec_qwen2vl_2b.yaml

    # With overrides
    python scripts/train.py configs/presets/vlm2vec_qwen2vl_2b.yaml \
        train.args.learning_rate=1e-4 train.args.output_dir=./output

    # Direct single GPU
    python scripts/train.py configs/presets/vlm2vec_qwen2vl_2b.yaml

    # Disable WandB for debugging
    python scripts/train.py configs/presets/vlm2vec_qwen2vl_2b.yaml
"""

import argparse
import logging
import sys

from vlm2emb.utils.logging import (
    DATE_FORMAT,
    LOG_FORMAT,
    add_rank_filter,
    set_verbosity,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BToks and VLM2Vec retrieval models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., training.lr=1e-4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with function names and line numbers",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")
    set_verbosity()

    # Add file handler if log file requested.
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        add_rank_filter(file_handler)
        logging.getLogger().addHandler(file_handler)

    logger.info("Loading config from: %s", args.config)

    # Import here to avoid slow startup for --help
    from omegaconf import OmegaConf

    from vlm2emb import train
    from vlm2emb.config import ConfigLoader, apply_overrides, to_native_config

    # Load config with interpolations unresolved so CLI overrides can still affect them.
    loader = ConfigLoader()
    config = loader.load_with_inheritance(str(args.config), resolve_interpolation=False)
    logger.info("Loaded config from: %s", args.config)

    # 2. Apply CLI overrides, then resolve interpolations.
    if args.overrides:
        config = apply_overrides(config, args.overrides)
        logger.info("Applied %d config override(s)", len(args.overrides))
    OmegaConf.resolve(config)

    # 3. Normalize runtime config boundary before orchestration
    runtime_config = to_native_config(config, resolve=True)

    # 4. Train (all logic in src/)
    train(runtime_config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise
