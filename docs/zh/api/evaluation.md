# Evaluation API 参考

> **状态**: Implemented
> **更新日期**: 2026-03-26
> **模块**: `vlm2emb.evaluation.evaluate`

本文档描述当前实现中的评测入口 API，而不是历史设计草案。

---

## 1. 核心入口

### 1.1 `evaluate`

```python
def evaluate(
    config,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
```

职责：

- 从运行时 `checkpoint` 或 `config["model"]` 解析模型来源
- 加载 model / processor
- 运行 benchmark
- 在主进程写入 `results.json` 与解析后的 `config.yaml`
- 执行同步与 runtime 收尾

参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `OmegaConf` 或 `dict` | 已加载的完整配置，必须包含 `eval` |
| `checkpoint` | `str \| None` | 运行时 checkpoint 路径；不属于配置语义 |
| `output_dir` | `str \| None` | 结果输出目录；为空时自动推导 |
| `config_path` | `str \| None` | 原始配置文件路径，仅用于结果元信息 |

返回：

| 类型 | 说明 |
|------|------|
| `dict[str, Any]` | benchmark 结果字典，包含各数据集结果与可选 `_summary` |

说明：

- `checkpoint` 优先于 `config["model"]`
- PEFT 场景下，processor 优先从 `base_model` 初始化，checkpoint 目录只作为回退
- `config` 中若出现 `checkpoint`，当前评测流程不会读取或使用

---

## 2. 运行时加载

### 2.1 `load_model_and_processor`

```python
def load_model_and_processor(
    config: Any,
    checkpoint: str | None = None,
) -> tuple[Any, Any, Accelerator]:
```

职责：

- 统一 standalone eval 的模型与 processor 解析逻辑
- 创建并返回本次评测使用的 `Accelerator`

行为：

| 场景 | 加载方式 |
|------|----------|
| full checkpoint | `BToks.from_pretrained(checkpoint)` |
| PEFT checkpoint | 优先 `AutoPeftModel.from_pretrained(checkpoint)`，失败时回退到 config 初始化 |
| 无 checkpoint | `create_model_and_processor(config)` |

---

## 3. Processor 解析规则

### 3.1 `full checkpoint`

优先级：

1. 从 checkpoint 目录加载 processor
2. 失败时，从 `config` 解析 processor

### 3.2 `PEFT checkpoint`

优先级：

1. 从 `base_model_name_or_path` 指向的 `base_model` 目录加载 processor
2. 若失败，再回退到 checkpoint 目录
3. 若 direct PEFT 加载失败且存在 `config["model"]`，才使用 config 初始化整模型

### 3.3 `config` fallback

`_load_processor_from_config(config)` 只做**纯配置解析**：

- 若存在 `config["processor"]`，直接走 `AutoProcessorWrapper.from_config(...)`
- 否则从 `config["model"].modules.*.model_name_or_path` 推断 backbone 路径
- 不会为了找 processor 而重建整模型

---

## 4. 生命周期与收尾

`evaluate(...)` 内部持有 `Accelerator`，并负责完整收尾：

1. benchmark 完成后 `accelerator.wait_for_everyone()`
2. 主进程写入结果
3. 再次 `accelerator.wait_for_everyone()`
4. 在 `finally` 中调用 `accelerator.end_training()`

说明：

- `scripts/eval.py` 不再负责 distributed lifecycle
- 收尾异常不会静默吞掉，而是记录 warning

---

## 5. CLI 入口关系

`scripts/eval.py` 当前只负责：

1. 解析参数
2. 读取配置
3. 应用 `overrides`
4. 调用 `evaluate(...)`

其中：

- `--checkpoint`、`--output-dir` 是运行时 flag
- `overrides` 继续保留为 dotlist
- 参数顺序通过 `parse_intermixed_args()` 放宽

---

## 6. 相关文档

- [评估系统设计](../architecture/evaluation-system.md)
- [配置系统](../architecture/config-system.md)
- [Model API](./model.md)
