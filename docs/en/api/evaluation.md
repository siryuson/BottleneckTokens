# Evaluation API Reference

> **Status**: Implemented
> **Updated**: 2026-03-26
> **Module**: `vlm2emb.evaluation.evaluate`

This document describes the current evaluation API, not the historical design draft.

---

## 1. Core Entry Point

### 1.1 `evaluate`

```python
def evaluate(
    config,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
```

Responsibilities:

- resolve the runtime model source from the explicit `checkpoint` argument or `config["model"]`
- load model / processor
- run the benchmark
- write `results.json` and resolved `config.yaml` on the main process
- synchronize and finalize the evaluation runtime

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `OmegaConf` or `dict` | Fully loaded config; must contain `eval` |
| `checkpoint` | `str \| None` | Runtime checkpoint path; not part of config semantics |
| `output_dir` | `str \| None` | Output directory for eval artifacts; auto-resolved when omitted |
| `config_path` | `str \| None` | Original config file path, stored only as metadata |

Returns:

| Type | Description |
|------|-------------|
| `dict[str, Any]` | Benchmark results, including per-dataset results and optional `_summary` |

Notes:

- `checkpoint` takes priority over `config["model"]`
- In PEFT mode, the processor is initialized from `base_model` first; checkpoint-local artifacts are only a fallback
- If `checkpoint` appears inside `config`, the current eval flow does not read or use it

---

## 2. Runtime Loading

### 2.1 `load_model_and_processor`

```python
def load_model_and_processor(
    config: Any,
    checkpoint: str | None = None,
) -> tuple[Any, Any, Accelerator]:
```

Responsibilities:

- unify model / processor resolution for standalone evaluation
- create and return the `Accelerator` used by the current eval run

Behavior:

| Scenario | Loading path |
|----------|--------------|
| full checkpoint | `BToks.from_pretrained(checkpoint)` |
| PEFT checkpoint | prefer `AutoPeftModel.from_pretrained(checkpoint)`, fall back to config initialization if needed |
| no checkpoint | `create_model_and_processor(config)` |

---

## 3. Processor Resolution Rules

### 3.1 Full checkpoint

Priority:

1. load processor from the checkpoint directory
2. if that fails, resolve processor from `config`

### 3.2 PEFT checkpoint

Priority:

1. load processor from `base_model_name_or_path`
2. fall back to the checkpoint directory only if needed
3. if direct PEFT loading fails and `config["model"]` exists, initialize the model from config

### 3.3 Config fallback

`_load_processor_from_config(config)` performs **pure config resolution** only:

- if `config["processor"]` exists, use `AutoProcessorWrapper.from_config(...)`
- otherwise infer the backbone path from `config["model"].modules.*.model_name_or_path`
- it does not rebuild the whole model just to discover the processor

---

## 4. Lifecycle and Finalization

`evaluate(...)` owns the full evaluation lifecycle:

1. `accelerator.wait_for_everyone()` after benchmark execution
2. main process writes results
3. another `accelerator.wait_for_everyone()`
4. `accelerator.end_training()` in `finally`

Notes:

- `scripts/eval.py` no longer manages the distributed lifecycle
- cleanup failures are logged as warnings instead of being silently swallowed

---

## 5. Relationship with the CLI Entry

`scripts/eval.py` now only:

1. parses arguments
2. loads config
3. applies `overrides`
4. calls `evaluate(...)`

Specifically:

- `--checkpoint` and `--output-dir` are runtime flags
- `overrides` remain dotlist-style inputs
- argument ordering is relaxed via `parse_intermixed_args()`

---

## 6. Related Documents

- [Evaluation System Design](../architecture/evaluation-system.md)
- [Configuration System](../architecture/config-system.md)
- [Model API](./model.md)
