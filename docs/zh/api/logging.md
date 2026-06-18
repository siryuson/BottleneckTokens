# Logging API 参考

> **版本**: 0.3
> **更新日期**: 2026-02-11
> **模块**: `vlm2emb.utils.logging`

本文档提供 BToks 日志系统 API 的详细参考。

---

## 目录

1. 概述
2. Logger 选择指南
3. [setup_logging](#setup_logging) - 初始化日志系统
4. [set_verbosity](#set_verbosity) - 控制第三方库日志级别
5. [set_level](#set_level) - 动态调整日志级别
6. 常量
7. 使用示例

---

## 概述

BToks 的日志系统遵循 Python 标准库 `logging` 最佳实践，明确区分库层与应用层职责：

- **库层**（`src/vlm2emb/**`）：仅使用标准 `logging.getLogger(__name__)`，无分布式运行时依赖。
- **应用层**（`scripts/**`、`src/vlm2emb/cli/**`）：在入口配置日志。可在运行时初始化后使用 `accelerate` 日志能力。

### 设计原则

1. **导入安全**：导入任何 `vlm2emb` 模块不会触发分布式初始化或运行时副作用。
2. **标准库优先（库层）**：库代码使用 `logging.getLogger(__name__)` — 无 accelerate 专用参数或封装。
3. **应用层控制配置**：日志 handler、formatter、level 仅在应用入口通过 `setup_logging()` 配置。
4. **Spawn 安全**：整个导入链对 `multiprocessing.start_method="spawn"` 安全。

---

## Logger 选择指南

| 场景 | 使用的 Logger |
|------|-------------|
| `src/vlm2emb/**`（不含 `cli/`） | `logging.getLogger(__name__)` |
| `src/vlm2emb/cli/**` | 标准 logging；初始化后可用 accelerate |
| `scripts/**` | 标准 logging；初始化后可用 accelerate |
| 模块顶层 / 全局变量 | `logging.getLogger(__name__)` — 禁止分布式调用 |
| 函数体内、运行时已初始化 | 按所在层级规则选择 |

### 规则

- 库模块 MUST NOT 导入 `accelerate.logging`。
- 库模块 MUST NOT 在日志调用中传递 accelerate 专用参数（`main_process_only`、`in_order`）。
- 库模块 MUST NOT 在模块级别注册 handler 或调用 `logging.basicConfig()`。
- 分布式日志过滤（如仅 rank-0 输出）是应用层职责。

---

## setup_logging

使用项目标准格式初始化根日志记录器。仅在应用入口调用。

### 函数签名

```python
def setup_logging(level: str = "INFO", verbose: bool = False) -> None
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | `str` | `"INFO"` | 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL） |
| `verbose` | `bool` | `False` | 是否使用详细格式（包含函数名和行号） |

### 行为

- 调用 `logging.basicConfig()` 并设置 `force=True` 配置根 logger。
- 输出到 `sys.stdout`。
- 库层 logger 通过层级传播自动输出。

### 示例

```python
from vlm2emb.utils.logging import setup_logging

# 基本初始化
setup_logging()

# 调试用详细模式
setup_logging(level="DEBUG", verbose=True)
```

---

## set_verbosity

设置第三方库的日志级别以减少噪音。

### 函数签名

```python
def set_verbosity(level: str = "WARNING") -> None
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | `str` | `"WARNING"` | 日志级别 |

### 影响的库

- `transformers`
- `datasets`
- `tokenizers`
- `accelerate`
- `urllib3`
- `filelock`
- `matplotlib`
- `PIL`

### 示例

```python
from vlm2emb.utils.logging import set_verbosity

# 仅显示警告和错误
set_verbosity("WARNING")

# 仅显示错误
set_verbosity("ERROR")
```

---

## set_level

动态调整根日志级别，无需重新初始化。

### 函数签名

```python
def set_level(level: str) -> None
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `level` | `str` | 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL） |

### 示例

```python
from vlm2emb.utils.logging import setup_logging, set_level

setup_logging(level="INFO")

# 运行一段时间后需要更多细节
set_level("DEBUG")
```

---

## 常量

| 常量 | 值 | 说明 |
|------|---|------|
| `LOG_FORMAT` | `"[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"` | 标准日志格式 |
| `LOG_FORMAT_VERBOSE` | `"[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"` | 详细格式 |
| `DATE_FORMAT` | `"%Y-%m-%d %H:%M:%S"` | 日期格式 |

---

## 使用示例

### 库模块

```python
"""src/vlm2emb/ 下的库模块示例。"""
import logging

logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Processing %d items", len(data))
    # ... 处理逻辑 ...
    logger.debug("Processing complete")
```

### 训练脚本（应用层）

```python
#!/usr/bin/env python3
"""训练脚本示例。"""
import logging

from vlm2emb.utils.logging import setup_logging, set_verbosity

# 1. 在入口配置日志
setup_logging()
set_verbosity("WARNING")

logger = logging.getLogger(__name__)

# 2. 导入其他模块
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

### 分布式日志过滤（应用层）

```python
#!/usr/bin/env python3
"""分布式训练中的 rank 感知日志。"""
import logging

from vlm2emb.utils.logging import setup_logging, set_verbosity

setup_logging()
set_verbosity("WARNING")

logger = logging.getLogger(__name__)

from accelerate import Accelerator

def main():
    accelerator = Accelerator()

    # 应用层 rank 过滤
    if accelerator.is_main_process:
        logger.info("Training started (main process only)")

    # 或使用 accelerate 的日志实现分布式感知
    from accelerate.logging import get_logger
    dist_logger = get_logger(__name__)
    dist_logger.info("This logs from main process only by default")

if __name__ == "__main__":
    main()
```

---

## 日志格式

### 标准格式

```
[2026-02-11 10:30:45] [INFO] [vlm2emb.training.train] Training started
```

### 详细格式（verbose=True）

```
[2026-02-11 10:30:45] [INFO] [vlm2emb.training.train:main:42] Training started
```

---

## 迁移指南

| 旧 API | 新 API |
|--------|--------|
| `get_logger(__name__)` (vlm2emb) | `logging.getLogger(__name__)`（标准库） |
| `logger.info("msg", main_process_only=False)` | `logger.info("msg")`（在应用层使用 accelerate 过滤） |
| `debug_print_all_ranks()` | 在应用层使用标准 logging + rank 检查 |
| `is_main_process()` | 在应用层使用 `accelerator.is_main_process` |
| `configure_logging()` | `setup_logging()` |
