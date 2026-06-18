# Contributing Guide

This guide covers contributing to BToks, coding standards, and testing practices.

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- CUDA toolkit (for GPU development)

### Installation

```bash
# Clone and install in editable mode
git clone https://github.com/siryuson/BottleneckTokens.git
cd BottleneckTokens
pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run linting
ruff check src/

# Run tests
pytest tests/
```

## Coding Standards

### Code Quality Tools

| Tool | Purpose | Config |
|------|---------|--------|
| Ruff | Linting, import sorting | `pyproject.toml` |
| Black | Formatting | line-length=100 |
| pytest | Testing | `pytest.ini` |

### Ruff Rules

- Line length: 100 characters max
- Rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade), B (bugbear)

```bash
# Check code
ruff check src/

# Auto-fix issues
ruff check --fix src/
```

### Type Hints

All public functions must have type hints:

```python
def load_config(path: str) -> dict:
    """Load configuration from YAML file.

    从 YAML 文件加载配置。

    Args:
        path: Path to YAML file / YAML 文件路径

    Returns:
        Configuration dictionary / 配置字典
    """
```

### Bilingual Documentation

All docstrings in `src/` must be bilingual (English + Chinese):

```python
# ✅ Correct
def forward(self, inputs: Tensor) -> Tensor:
    """Process inputs through the model.

    将输入通过模型处理。
    """

# ❌ Wrong - English only
def forward(self, inputs: Tensor) -> Tensor:
    """Process inputs through the model."""
```

## Adding Components

### Register a New Backbone Module

```python
# src/vlm2emb/modules/my_backbone.py
import torch.nn as nn
from transformers import AutoModel
from vlm2emb import AutoModule

@AutoModule.register("MyBackbone")
class MyBackbone(nn.Module):
    """Custom backbone implementation.

    自定义骨干网络实现。
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, **kwargs)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": attention_mask,
        }
```

### Register a New Pooling Module

```python
# src/vlm2emb/modules/my_pooling.py
import torch
import torch.nn as nn
from vlm2emb import AutoModule

@AutoModule.register("MyPooling")
class MyPooling(nn.Module):
    """Custom pooling module.

    自定义池化模块。
    """

    def forward(self, last_hidden_state, attention_mask=None, **kwargs):
        if attention_mask is None:
            embeddings = last_hidden_state.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return {"embeddings": embeddings, **kwargs}
```

### Register a New Dataset

```python
# src/vlm2emb/data/datasets/my_dataset.py
from vlm2emb import AutoDataset

@AutoDataset.register("my_dataset")
def load_my_dataset(path: str, split: str = "train", **kwargs):
    """Load custom dataset.

    加载自定义数据集。
    """
    from datasets import load_dataset
    return load_dataset(path, split=split, **kwargs)
```

## Testing

### Test Structure

```
tests/
├── data/              # Data module tests
├── evaluate/          # Evaluation module tests
├── grad_cache/        # GradCache tests
├── integration/       # Integration tests
├── loss/              # Loss function tests
├── models/            # Model related tests
├── trainers/          # Trainer tests
├── utils/             # Utility tests
└── conftest.py         # Shared fixtures
```

### Test Markers

```python
import pytest

@pytest.mark.unit
def test_config_loading():
    """Unit test for config loading."""
    pass

@pytest.mark.integration
def test_training_pipeline():
    """Integration test for training."""
    pass

@pytest.mark.slow
def test_full_training():
    """Slow test (excluded by default)."""
    pass
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest -m unit tests/

# Run with coverage
pytest --cov=src/vlm2emb tests/

# Run specific test file
pytest tests/test_config_loader.py
```

## Exception Handling

Use project-specific exceptions:

```python
from vlm2emb.exceptions import VLM2EmbError, ConfigError, ComponentError

# ✅ Correct
raise ConfigError(f"Invalid config: {details}")

# ❌ Wrong
raise ValueError(f"Invalid config: {details}")
```

### Exception Hierarchy

```
VLM2EmbError (base)
├── ConfigError
│   ├── ConfigNotFoundError
│   ├── ConfigSyntaxError
│   └── ConfigValidationError
├── ComponentError
│   ├── ComponentNotFoundError
│   └── InterfaceValidationError
└── DataError
    └── DataLoadError
```

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `Qwen2VLBackbone` |
| Functions | snake_case | `load_config` |
| Constants | UPPER_SNAKE | `DEFAULT_BATCH_SIZE` |
| Files | snake_case.py | `qwen2vl_backbone.py` |
| Configs | snake_case.yaml | `qwen2vl_7b.yaml` |

**Backbone version naming**:
- Qwen2-VL → `qwen2vl_{size}` (e.g., `qwen2vl_7b`)
- Qwen2.5-VL → `qwen2_5vl_{size}` (underscore for decimals)

---

_For architecture details, see [Architecture Overview](../architecture/overview.md)._
