# Logging API Reference

> **Version**: 0.3
> **Updated**: 2026-02-11
> **Module**: `vlm2emb.utils.logging`

This document provides a detailed reference for the BToks logging system API.

---

## Table of Contents

1. [Overview](#overview)
2. [Logger Selection Guide](#logger-selection-guide)
3. [setup_logging](#setup_logging) - Initialize logging system
4. [set_verbosity](#set_verbosity) - Control third-party library log levels
5. [set_level](#set_level) - Dynamically adjust log level
6. [Constants](#constants)
7. [Usage Examples](#usage-examples)

---

## Overview

BToks's logging system follows Python standard library `logging` best practices with a clear separation between library-layer and application-layer concerns:

- **Library layer** (`src/vlm2emb/**`): Uses standard `logging.getLogger(__name__)` only. No distributed runtime dependencies.
- **Application layer** (`scripts/**`, `src/vlm2emb/cli/**`): Configures logging at entry point. May use `accelerate` logging capabilities after runtime initialization.

### Design Principles

1. **Import-safe**: Importing any `vlm2emb` module never triggers distributed initialization or runtime side effects.
2. **Standard library only (library layer)**: Library code uses `logging.getLogger(__name__)` — no accelerate-specific parameters or wrappers.
3. **Application-controlled configuration**: Log handlers, formatters, and levels are configured only at application entry points via `setup_logging()`.
4. **Spawn-safe**: The entire import chain is safe for `multiprocessing.start_method="spawn"`.

---

## Logger Selection Guide

| Context | Logger to Use |
|---------|--------------|
| `src/vlm2emb/**` (excluding `cli/`) | `logging.getLogger(__name__)` |
| `src/vlm2emb/cli/**` | Standard logging; may use accelerate after init |
| `scripts/**` | Standard logging; may use accelerate after init |
| Module top-level / global variables | `logging.getLogger(__name__)` — no distributed calls |
| Inside function body, runtime initialized | Follow layer rules above |

### Rules

- Library modules MUST NOT import `accelerate.logging`.
- Library modules MUST NOT pass accelerate-specific parameters (`main_process_only`, `in_order`) to log calls.
- Library modules MUST NOT register handlers or call `logging.basicConfig()` at module level.
- Distributed log filtering (e.g., rank-0 only) is an application-layer concern.

---

## setup_logging

Initialize the root logger with project-standard format. Called at application entry points only.

### Function Signature

```python
def setup_logging(level: str = "INFO", verbose: bool = False) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `str` | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `verbose` | `bool` | `False` | Use verbose format (includes function name and line number) |

### Behavior

- Calls `logging.basicConfig()` with `force=True` to configure the root logger.
- Output goes to `sys.stdout`.
- Library-layer loggers propagate to root automatically.

### Example

```python
from vlm2emb.utils.logging import setup_logging

# Basic initialization
setup_logging()

# Verbose mode for debugging
setup_logging(level="DEBUG", verbose=True)
```

---

## set_verbosity

Set log level for third-party libraries to reduce noise.

### Function Signature

```python
def set_verbosity(level: str = "WARNING") -> None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `str` | `"WARNING"` | Log level |

### Affected Libraries

- `transformers`
- `datasets`
- `tokenizers`
- `accelerate`
- `urllib3`
- `filelock`
- `matplotlib`
- `PIL`

### Example

```python
from vlm2emb.utils.logging import set_verbosity

# Show only warnings and errors
set_verbosity("WARNING")

# Show only errors
set_verbosity("ERROR")
```

---

## set_level

Dynamically adjust root log level without reinitializing.

### Function Signature

```python
def set_level(level: str) -> None
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `str` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Example

```python
from vlm2emb.utils.logging import setup_logging, set_level

setup_logging(level="INFO")

# Later, need more detail
set_level("DEBUG")
```

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `LOG_FORMAT` | `"[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"` | Standard log format |
| `LOG_FORMAT_VERBOSE` | `"[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"` | Verbose format |
| `DATE_FORMAT` | `"%Y-%m-%d %H:%M:%S"` | Date format |

---

## Usage Examples

### Library Module

```python
"""Example library module in src/vlm2emb/."""
import logging

logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Processing %d items", len(data))
    # ... processing logic ...
    logger.debug("Processing complete")
```

### Training Script (Application Layer)

```python
#!/usr/bin/env python3
"""Training script example."""
import logging

from vlm2emb.utils.logging import setup_logging, set_verbosity

# 1. Configure logging at entry point
setup_logging()
set_verbosity("WARNING")

logger = logging.getLogger(__name__)

# 2. Import other modules
from vlm2emb import train
from vlm2emb.config import load_config

def main():
    logger.info("Training started")
    config = load_config("config.yaml")
    train(config)
    logger.info("Training complete")

if __name__ == "__main__":
    main()
```

### Distributed Log Filtering (Application Layer)

```python
#!/usr/bin/env python3
"""Distributed training with rank-aware logging."""
import logging

from vlm2emb.utils.logging import setup_logging, set_verbosity

setup_logging()
set_verbosity("WARNING")

logger = logging.getLogger(__name__)

from accelerate import Accelerator

def main():
    accelerator = Accelerator()

    # Application-layer rank filtering
    if accelerator.is_main_process:
        logger.info("Training started (main process only)")

    # Or use accelerate's logging for distributed awareness
    from accelerate.logging import get_logger
    dist_logger = get_logger(__name__)
    dist_logger.info("This logs from main process only by default")

if __name__ == "__main__":
    main()
```

---

## Log Format

### Standard Format

```
[2026-02-11 10:30:45] [INFO] [vlm2emb.training.train] Training started
```

### Verbose Format (verbose=True)

```
[2026-02-11 10:30:45] [INFO] [vlm2emb.training.train:main:42] Training started
```

---

## Migration Guide

| Old API | New API |
|---------|---------|
| `get_logger(__name__)` from vlm2emb | `logging.getLogger(__name__)` (standard library) |
| `logger.info("msg", main_process_only=False)` | `logger.info("msg")` (use accelerate at app layer for filtering) |
| `debug_print_all_ranks()` | Use standard logging + rank checks at app layer |
| `is_main_process()` | `accelerator.is_main_process` at app layer |
| `configure_logging()` | `setup_logging()` |
