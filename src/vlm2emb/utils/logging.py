"""Logging utilities for BToks public runtime entry points.

This module provides logging configuration utilities for application entry points.
Distributed runtime code should use `RankLogger(__name__)` unless it has a
specific reason to manage ranks manually.

Usage:
    from vlm2emb.utils.logging import RankLogger, setup_logging, set_verbosity

    setup_logging()
    set_verbosity("WARNING")

    logger = RankLogger(__name__)

    logger.info("Only rank 0 sees this")
    logger.info("All ranks in order", ranks=[-1], in_order=True)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

# Log format constants.
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_FORMAT_VERBOSE = (
    "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Thread-local flag: set by log_ranks to bypass _RankFilter
_log_ranks_active = threading.local()


class _RankFilter(logging.Filter):
    """Suppress uncontrolled log records on non-rank-0 processes.

    Passes through:
    - All records from log_ranks (already rank-controlled, marked via thread-local)
    - All records on rank 0
    - All records in non-distributed environment
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(_log_ranks_active, "active", False):
            return True
        rank_info = _get_rank_info()
        if rank_info is None:
            return True
        rank, _ = rank_info
        return rank == 0


def _parse_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_env_rank_info() -> tuple[int, int] | None:
    """Read rank info from launcher environment before frameworks initialize."""

    rank_world_pairs = (
        ("RANK", "WORLD_SIZE"),
        ("ACCELERATE_PROCESS_INDEX", "ACCELERATE_NUM_PROCESSES"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),
    )
    for rank_name, world_name in rank_world_pairs:
        rank = _parse_env_int(rank_name)
        world_size = _parse_env_int(world_name)
        if rank is not None and world_size is not None and world_size > 1:
            return rank, world_size

    # Last resort for launchers that expose only local rank. This may allow one
    # logger per node in multi-node jobs, so global rank envs above take priority.
    local_rank = _parse_env_int("LOCAL_RANK")
    local_world_size = _parse_env_int("LOCAL_WORLD_SIZE")
    if local_rank is not None and local_world_size is not None and local_world_size > 1:
        return local_rank, local_world_size

    return None


def _get_rank_info() -> tuple[int, int] | None:
    """Get current rank and world size without triggering initialization.

    Returns:
        (rank, world_size) tuple, or None if not in a distributed environment.
    """
    # Priority 1: accelerate PartialState (safe check, no init)
    try:
        from accelerate.state import PartialState

        if PartialState._shared_state != {}:
            state = PartialState()
            if state.num_processes > 1:
                return state.process_index, state.num_processes
    except ImportError:
        pass

    # Priority 2: torch.distributed
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                return dist.get_rank(), world_size
    except ImportError:
        pass

    env_rank_info = _get_env_rank_info()
    if env_rank_info is not None:
        return env_rank_info

    # Not in distributed environment
    return None


def _wait_for_everyone() -> None:
    """Execute a distributed barrier using the same priority as _get_rank_info."""
    try:
        from accelerate.state import PartialState

        if PartialState._shared_state != {}:
            state = PartialState()
            if state.num_processes > 1:
                state.wait_for_everyone()
                return
    except ImportError:
        pass

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            if dist.get_world_size() > 1:
                dist.barrier()
                return
    except ImportError:
        pass


def log_ranks(
    logger: logging.Logger,
    msg: str,
    *args: Any,
    level: int = logging.INFO,
    ranks: list[int] | None = None,
    in_order: bool = False,
    exc_info: Any = None,
) -> None:
    """Log a message with explicit rank control.

    Args:
        logger: Standard logging.Logger instance.
        msg: Log message (supports %s formatting).
        *args: Format arguments for msg.
        level: Log level (default INFO). Keyword-only.
        ranks: List of ranks that should log. Keyword-only.
               [0] (default) = only rank 0.
               [-1] = all ranks.
               [0, 2] = only ranks 0 and 2.
        in_order: If True, log sequentially by rank order using barriers.
                  Keyword-only. Default False.
                  Only meaningful when multiple ranks are logging.

    Examples:
        >>> log_ranks(logger, "step %d loss %.4f", step, loss)
        >>> log_ranks(logger, "all ranks", ranks=[-1])
        >>> log_ranks(logger, "ordered", ranks=[-1], in_order=True)
        >>> log_ranks(logger, "debug info", level=logging.DEBUG)
    """
    if ranks is None:
        ranks = [0]

    rank_info = _get_rank_info()

    # Non-distributed: log unconditionally
    if rank_info is None:
        logger.log(level, msg, *args, exc_info=exc_info)
        return

    rank, world_size = rank_info
    all_ranks = -1 in ranks
    should_log = all_ranks or rank in ranks

    # Prepend [rank{N}] in distributed environment (D6)
    if should_log:
        prefixed_msg = f"[rank{rank}] {msg}"
    else:
        prefixed_msg = msg

    # Single rank logging or no ordering: simple path
    if not in_order or (not all_ranks and len(ranks) <= 1):
        if should_log:
            _log_ranks_active.active = True
            try:
                logger.log(level, prefixed_msg, *args, exc_info=exc_info)
            finally:
                _log_ranks_active.active = False
        return

    # Ordered logging: all processes participate in barriers
    _wait_for_everyone()
    for i in range(world_size):
        if i == rank and should_log:
            _log_ranks_active.active = True
            try:
                logger.log(level, prefixed_msg, *args, exc_info=exc_info)
            finally:
                _log_ranks_active.active = False
        _wait_for_everyone()


def add_rank_filter(handler: logging.Handler) -> None:
    """Attach the project rank filter to a logging handler."""

    if not any(isinstance(existing, _RankFilter) for existing in handler.filters):
        handler.addFilter(_RankFilter())


def setup_logging(
    level: str = "INFO",
    verbose: bool = False,
) -> None:
    """Configure root logger with project-standard format.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        verbose: If True, use verbose format with function name and line number.
    """
    fmt = LOG_FORMAT_VERBOSE if verbose else LOG_FORMAT
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Add rank filter to root handlers. Explicit log_ranks calls bypass this
    # through the thread-local flag after deciding which ranks should log.
    for handler in logging.root.handlers:
        add_rank_filter(handler)


def set_verbosity(level: str = "WARNING") -> None:
    """Set verbosity level for third-party libraries.

    Controls logging output from transformers, datasets, and other libraries.
    Useful for reducing noise during training.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Example:
        >>> set_verbosity("WARNING")  # Only warnings and errors
        >>> set_verbosity("ERROR")    # Only errors
    """
    log_level = getattr(logging, level.upper(), logging.WARNING)

    # HuggingFace transformers
    try:
        import transformers.utils.logging as hf_logging

        hf_logging.set_verbosity(log_level)
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except ImportError:
        pass

    # HuggingFace datasets
    try:
        import datasets.utils.logging as ds_logging

        ds_logging.set_verbosity(log_level)
    except ImportError:
        pass

    # Other noisy libraries.
    noisy_loggers = [
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "urllib3",
        "filelock",
        "matplotlib",
        "PIL",
        "databus",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(log_level)

def set_level(level: str) -> None:
    """Dynamically change the root log level.

    This allows changing the log level without reinitializing the logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Example:
        >>> setup_logging(level="INFO")
        >>> # ... later, need more detail ...
        >>> set_level("DEBUG")
    """
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


class RankLogger:
    """Rank-aware logger wrapper providing standard logging API.

    Wraps ``log_ranks()`` with standard convenience methods. Default rank
    filtering:

    - ``debug`` / ``info`` / ``warning`` / ``error`` / ``exception``: rank 0 only
    - pass ``ranks=[-1]`` when a call site intentionally needs all-rank output

    Example::

        logger = RankLogger(__name__)
        logger.info("step %d", step)
        logger.warning("empty GT detected")
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, msg: str, *args: Any, ranks: list[int] | None = None, in_order: bool = False) -> None:
        log_ranks(self._logger, msg, *args, level=logging.INFO, ranks=ranks, in_order=in_order)

    def debug(self, msg: str, *args: Any, ranks: list[int] | None = None, in_order: bool = False) -> None:
        log_ranks(self._logger, msg, *args, level=logging.DEBUG, ranks=ranks, in_order=in_order)

    def warning(self, msg: str, *args: Any, ranks: list[int] | None = None, in_order: bool = False) -> None:
        log_ranks(self._logger, msg, *args, level=logging.WARNING, ranks=ranks, in_order=in_order)

    def error(self, msg: str, *args: Any, ranks: list[int] | None = None, in_order: bool = False) -> None:
        log_ranks(self._logger, msg, *args, level=logging.ERROR, ranks=ranks, in_order=in_order)

    def exception(self, msg: str, *args: Any, ranks: list[int] | None = None, in_order: bool = False) -> None:
        log_ranks(
            self._logger,
            msg,
            *args,
            level=logging.ERROR,
            ranks=ranks,
            in_order=in_order,
            exc_info=True,
        )


# Convenience exports.
__all__ = [
    "setup_logging",
    "set_verbosity",
    "set_level",
    "add_rank_filter",
    "log_ranks",
    "RankLogger",
    "LOG_FORMAT",
    "LOG_FORMAT_VERBOSE",
    "DATE_FORMAT",
]
