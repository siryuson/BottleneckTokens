#!/usr/bin/env python3
"""Evaluate BToks and VLM2Vec retrieval models.

Usage:
    # Direct single GPU
    python scripts/eval.py configs/presets/vlm2vec_qwen2vl_2b.yaml

    # Evaluate a checkpoint
    python scripts/eval.py configs/presets/vlm2vec_qwen2vl_2b.yaml \
        --checkpoint /path/to/checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

from vlm2emb.utils.logging import (
    DATE_FORMAT,
    LOG_FORMAT,
    RankLogger,
    add_rank_filter,
    set_verbosity,
    setup_logging,
)

logger = RankLogger(__name__)


def _prepare_log_file(path: str) -> Path:
    """Ensure the log file parent directory exists before opening it."""
    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate BToks and VLM2Vec retrieval models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to preset config YAML (must contain eval: key)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., eval.batch_size=8 eval.datasets=[MSCOCO_i2t])",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or Hugging Face repo id",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save eval artifacts",
    )
    parse_intermixed = getattr(parser, "parse_intermixed_args", None)
    if parse_intermixed is not None:
        return parse_intermixed()
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")
    set_verbosity()

    # Add file handler if log file requested.
    if args.log_file:
        log_path = _prepare_log_file(args.log_file)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        add_rank_filter(file_handler)
        logging.getLogger().addHandler(file_handler)

    logger.info("Loading config from: %s", args.config)

    # Import here to avoid slow startup for --help
    from omegaconf import OmegaConf

    from vlm2emb.config import ConfigLoader, apply_overrides, to_native_config
    from vlm2emb.evaluation.evaluate import evaluate

    # 1. Load config (keep interpolations unresolved so CLI overrides take effect).
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

    # 4. Evaluate (all logic in src/)
    evaluate(
        runtime_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Evaluation failed with error: %s", e)
        raise
