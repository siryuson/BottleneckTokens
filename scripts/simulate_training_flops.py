#!/usr/bin/env python
"""Simulate training FLOPs from production configs without training."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from vlm2emb.analysis.training_flops import (
    DEFAULT_CONFIGS,
    SimulationOptions,
    aggregate_windows,
    format_windows,
    simulate_config,
    summarize_total,
    write_jsonl,
)
from vlm2emb.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Offline training FLOPs simulator for BToks/VLM2Vec configs.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        default=list(DEFAULT_CONFIGS),
        help="Config path(s). Defaults to VLM2Vec and BToks Qwen2-VL 2B presets.",
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--window-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--drop-last", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--train-forward-backward-multiplier", type=float, default=3.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument(
        "--exact-processor",
        action="store_true",
        help="Use the full processor path, including pixel preprocessing. Much slower.",
    )
    parser.add_argument(
        "--cache-source-shapes",
        action="store_true",
        help=(
            "Cache the first observed batch shape for each source dataset while "
            "still scanning the sampler source sequence."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotlist format. Can be passed multiple times.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _drop_last_value(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def main() -> None:
    """Run the simulator."""
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")
    options = SimulationOptions(
        max_steps=args.max_steps,
        window_steps=args.window_steps,
        batch_size=args.batch_size,
        world_size=args.world_size,
        rank=args.rank,
        drop_last=_drop_last_value(args.drop_last),
        train_forward_backward_multiplier=args.train_forward_backward_multiplier,
        progress_every=args.progress_every,
        overrides=tuple(args.override),
        exact_processor=args.exact_processor,
        cache_source_shapes=args.cache_source_shapes,
    )

    all_windows = []
    for config_path in args.configs:
        logger.info("Simulating config: %s", config_path)
        steps = simulate_config(config_path, options=options)
        windows = aggregate_windows(steps, args.window_steps)
        all_windows.extend(windows)
        summary = summarize_total(steps)
        logger.info(
            "%s total: steps=%d global_TFLOPs=%.6f per_device_TFLOPs=%.6f",
            Path(config_path).stem,
            int(summary["steps"]),
            summary["global_tflops"],
            summary["per_device_tflops"],
        )

    print(format_windows(all_windows))

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, all_windows)
        logger.info("Wrote window records to %s", args.output_jsonl)

    for window in all_windows:
        logger.debug("window=%s", asdict(window))


if __name__ == "__main__":
    main()
