# 开发指南

本指南涵盖为 BToks 做贡献、编码规范和测试实践。

## 开发环境设置

### 前置条件

- Python 3.11+
- Git
- CUDA 工具包（用于 GPU 开发）

### 安装

```bash
# 克隆并以可编辑模式安装
git clone https://github.com/siryuson/BottleneckTokens.git
cd BottleneckTokens
pip install -e ".[dev]"
```

### 验证设置

```bash
# 运行 lint
ruff check src/

# 运行测试
pytest tests/
```

## 编码规范

### 代码质量工具

| 工具 | 用途 | 配置 |
|------|------|------|
| Ruff | Lint、导入排序 | `pyproject.toml` |
| Black | 格式化 | line-length=100 |
| pytest | 测试 | `pytest.ini` |

### Ruff 规则

- 行长度：最多 100 字符
- 规则：E（pycodestyle）、F（pyflakes）、I（isort）、UP（pyupgrade）、B（bugbear）

```bash
# 检查代码
ruff check src/

# 自动修复问题
ruff check --fix src/
```

### 类型提示

所有公共函数必须有类型提示：

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

## 添加组件

### 注册新的 Backbone 模块

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

### 注册新的 Pooling 模块

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

### 注册新的数据集

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

## 测试

### 测试结构

```
tests/
├── data/              # 数据模块测试
├── evaluate/          # 评估模块测试
├── grad_cache/        # GradCache 测试
├── integration/       # 集成测试
├── loss/              # 损失函数测试
├── models/            # 模型相关测试
├── trainers/          # 训练器测试
├── utils/             # 工具测试
└── conftest.py         # 共享 fixtures
```

### 测试标记

```python
import pytest

@pytest.mark.unit
def test_config_loading():
    """配置加载的单元测试。"""
    pass

@pytest.mark.integration
def test_training_pipeline():
    """训练的集成测试。"""
    pass

@pytest.mark.slow
def test_full_training():
    """慢速测试（默认排除）。"""
    pass
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 仅运行单元测试
pytest -m unit tests/

# 带覆盖率运行
pytest --cov=src/vlm2emb tests/

# 运行特定测试文件
pytest tests/test_config_loader.py
```

## 异常处理

使用项目特定的异常：

```python
from vlm2emb.exceptions import VLM2EmbError, ConfigError, ComponentError

# ✅ 正确
raise ConfigError(f"Invalid config: {details}")

# ❌ 错误
raise ValueError(f"Invalid config: {details}")
```

### 异常层次结构

```
VLM2EmbError (基类)
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

## 命名规范

| 元素 | 规范 | 示例 |
|------|------|------|
| 类 | PascalCase | `Qwen2VLBackbone` |
| 函数 | snake_case | `load_config` |
| 常量 | UPPER_SNAKE | `DEFAULT_BATCH_SIZE` |
| 文件 | snake_case.py | `qwen2vl_backbone.py` |
| 配置 | snake_case.yaml | `qwen2vl_7b.yaml` |

**Backbone 版本命名：**
- Qwen2-VL → `qwen2vl_{size}`（如 `qwen2vl_7b`）
- Qwen2.5-VL → `qwen2_5vl_{size}`（小数用下划线）

---

_架构详情请参见 [架构概览](../architecture/overview.md)。_
