# 注册表系统

> **版本**: 0.1
> **更新日期**: 2026-01-25
> **架构设计版本**: 1.0
> **参考**: HuggingFace transformers Auto Classes

## 1. 设计理念

### 1.1 为什么需要注册表

BToks 采用注册表模式管理所有可插拔组件，实现：

| 目标 | 说明 |
|------|------|
| **配置驱动** | 通过 YAML 中的 `type` 字段指定组件，无需硬编码 |
| **动态发现** | 运行时自动发现已注册组件 |
| **易于扩展** | 新组件只需装饰器注册，无需修改框架代码 |
| **类型安全** | 可选的基类验证，确保组件接口一致 |

### 1.2 与 HuggingFace Auto Classes 对比

BToks 的注册表设计参考 HuggingFace transformers 的 Auto Classes：

```python
# HuggingFace 风格
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# BToks 风格
from vlm2emb import AutoModule
backbone = AutoModule.from_config({"type": "Qwen2VL", ...})
```

**主要差异**：

| 方面 | HuggingFace | BToks |
|------|-------------|---------|
| 组件发现 | 基于 model_type 映射 | 基于显式注册 |
| 实例化 | `from_pretrained()` | `from_config()` / `build()` |
| 注册方式 | 内部映射表 | 装饰器 / 直接注册 |

---

## 2. Registry 基类

### 2.1 类定义

```python
from vlm2emb import Registry

class Registry:
    """通用组件注册表。

    Attributes:
        name: 注册表名称（如 "backbone", "dataset"）
        base_class: 可选的基类，用于验证注册组件
        _components: 名称到组件的内部映射
    """

    def __init__(
        self,
        name: str,
        base_class: type | None = None,
        allow_override: bool = False,
    ):
        ...
```

### 2.2 核心方法

#### register() - 注册组件

```python
def register(
    self,
    name: str | None = None,
    component: type | Callable | None = None,
    override: bool = False,
) -> type | Callable:
    """注册组件类或工厂函数。

    支持两种模式：
    1. 装饰器：@Registry.register("name")
    2. 直接调用：Registry.register("name", Component)
    """
```

**使用示例**：

```python
from vlm2emb import AutoModule

# 模式 1：装饰器 + 指定名称
@AutoModule.register("MyPooling")
class MyPoolingModule:
    pass

# 模式 2：装饰器 + 自动使用类名
@AutoModule.register()
class WeightedPooling:
    pass  # 注册为 "WeightedPooling"

# 模式 3：直接注册
AutoModule.register("CustomModule", CustomModuleClass)
```

#### from_config() - 从配置实例化

```python
def from_config(self, config: dict, **kwargs) -> Any:
    """从配置字典实例化组件。

    Args:
        config: 必须包含 'type' 键的配置字典
        **kwargs: 覆盖配置的额外参数

    Returns:
        实例化的组件
    """
```

**使用示例**：

```python
from vlm2emb import AutoModule

# 从配置创建
backbone = AutoModule.from_config({
    "type": "Qwen2VLBackbone",
    "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
    "dtype": "bfloat16",
})

# 覆盖配置参数
backbone = AutoModule.from_config(
    {"type": "Qwen2VLBackbone", "model_name_or_path": "..."},
    dtype="float16",  # 覆盖 dtype
)
```

#### build() - 直接构建

```python
def build(self, name: str, *args, **kwargs) -> Any:
    """通过名称直接构建组件实例。"""
```

**使用示例**：

```python
from vlm2emb import AutoModule

# 直接构建
pooling = AutoModule.build("LastTokenPooling")
normalize = AutoModule.build("Normalize")
```

#### 其他方法

| 方法 | 说明 |
|------|------|
| `get(name)` | 获取已注册的组件类（不实例化） |
| `list_modules()` | 列出所有已注册组件名称 |
| `is_registered(name)` | 检查组件是否已注册 |
| `__contains__(name)` | 支持 `in` 操作符 |
| `__len__()` | 返回已注册组件数量 |

---

## 3. Auto* 预定义注册表

BToks 提供以下预定义注册表：

### 3.1 模型相关

| 注册表 | 说明 | 典型组件 |
|--------|------|----------|
| `AutoModel` | 完整模型 | BToks |
| `AutoModule` | 管道模块 | Qwen2VLBackbone, LastTokenPooling, Normalize |

### 3.2 训练相关

| 注册表 | 说明 | 典型组件 |
|--------|------|----------|
| `AutoTrainer` | 训练器 | VLM2VecTrainer |
| `AutoLoss` | 损失函数 | InfoNCELoss, ContrastiveLoss |
| `AutoOptimizer` | 优化器 | AdamW |
| `AutoScheduler` | 学习率调度器 | CosineScheduler |

### 3.3 数据相关

| 注册表 | 说明 | 典型组件 |
|--------|------|----------|
| `AutoDataset` | 数据集 | MMEBDataset, ViDoReDataset |
| `AutoCollator` | 数据整理器 | TrainingCollator |
| `AutoSampler` | 数据采样器 | DistributedSampler |

### 3.4 评估相关

| 注册表 | 说明 | 典型组件 |
|--------|------|----------|
| `AutoMetric` | 评估指标 | Recall, NDCG |
| `AutoEvaluator` | 评估器 | MMEBEvaluator |

### 3.5 使用示例

```python
from vlm2emb import (
    AutoModule,
    AutoLoss,
    AutoDataset,
)

# 列出可用组件
print(AutoModule.list_modules())
# ['Qwen2VLBackbone', 'LastTokenPooling', 'MeanPooling', 'Normalize', ...]

# 检查组件是否存在
if "Qwen2VLBackbone" in AutoModule:
    backbone = AutoModule.build("Qwen2VLBackbone", ...)
```

---

## 4. 组件注册

### 4.1 装饰器注册（推荐）

最常用的注册方式，在类定义时自动注册：

```python
from vlm2emb import AutoModule
import torch.nn as nn

@AutoModule.register("WeightedPooling")
class WeightedPooling(nn.Module):
    """自定义加权池化模块。"""

    def __init__(self, hidden_size: int = 3584):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, last_hidden_states, attention_mask=None, **kwargs):
        weighted = last_hidden_states * self.weight
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (weighted * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = weighted.mean(dim=1)
        return {"embeddings": pooled, **kwargs}
```

### 4.2 工厂函数注册

对于需要复杂初始化逻辑的组件，可以注册工厂函数：

```python
from vlm2emb import AutoDataset

@AutoDataset.register("custom_dataset")
def load_custom_dataset(data_path: str, split: str = "train", **kwargs):
    """工厂函数：加载自定义数据集。"""
    from datasets import load_dataset
    return load_dataset(data_path, split=split, **kwargs)
```

### 4.3 直接注册

适用于注册第三方类或动态注册：

```python
from vlm2emb import AutoLoss
from some_library import CustomLoss

# 直接注册第三方类
AutoLoss.register("CustomLoss", CustomLoss)

# 动态注册
def register_losses():
    for name, cls in discover_loss_classes():
        AutoLoss.register(name, cls)
```

### 4.4 带基类验证的注册

创建带基类验证的自定义注册表：

```python
from vlm2emb import Registry
import torch.nn as nn

# 创建要求继承 nn.Module 的注册表
ModuleRegistry = Registry("custom_modules", base_class=nn.Module)

@ModuleRegistry.register("ValidModule")
class ValidModule(nn.Module):  # ✅ 通过验证
    pass

@ModuleRegistry.register("InvalidModule")
class InvalidModule:  # ❌ TypeError: 必须继承 nn.Module
    pass
```

---

## 5. 配置集成

### 5.1 YAML 配置中的 type 字段

注册表与配置系统紧密集成，通过 `type` 字段指定组件：

```yaml
# configs/models/vlm2vec.yaml
model:
  type: vlm2emb  # AutoModel 查找

  modules:
    - type: Qwen2VLBackbone  # AutoModule 查找
      model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
      dtype: bfloat16

    - type: LastTokenPooling  # AutoModule 查找

    - type: Normalize  # AutoModule 查找

training:
  loss:
    type: InfoNCELoss  # AutoLoss 查找
    temperature: 0.02
```

### 5.2 from_config 与 from_config 类方法

如果组件类定义了 `from_config` 类方法，Registry 会优先使用它：

```python
@AutoModule.register("ComplexModule")
class ComplexModule(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        ...

    @classmethod
    def from_config(cls, config: dict) -> "ComplexModule":
        """自定义配置解析逻辑。"""
        config = config.copy()
        config.pop("type", None)

        # 自定义逻辑：根据 model_size 设置参数
        if "model_size" in config:
            size = config.pop("model_size")
            if size == "small":
                config["hidden_size"] = 768
                config["num_layers"] = 6
            elif size == "large":
                config["hidden_size"] = 1024
                config["num_layers"] = 12

        return cls(**config)
```

```yaml
# 使用自定义 from_config
modules:
  - type: ComplexModule
    model_size: large  # from_config 会转换为 hidden_size=1024, num_layers=12
```

---

## 6. 工具函数

### 6.1 get_registry()

通过名称获取预定义注册表：

```python
from vlm2emb import get_registry

backbone_registry = get_registry("backbone")
print(backbone_registry.list_modules())
```

### 6.2 list_registries()

列出所有可用注册表：

```python
from vlm2emb import list_registries

print(list_registries())
# ['backbone', 'head', 'model', 'module', 'trainer', 'loss', ...]
```

---

## 7. 自定义注册表

### 7.1 创建自定义注册表

```python
from vlm2emb import Registry

# 创建自定义注册表
MyCallbackRegistry = Registry("callback")

@MyCallbackRegistry.register("LoggingCallback")
class LoggingCallback:
    def on_step_end(self, step, loss):
        print(f"Step {step}: loss={loss:.4f}")

@MyCallbackRegistry.register("CheckpointCallback")
class CheckpointCallback:
    def on_epoch_end(self, epoch, model):
        model.save_pretrained(f"checkpoint-{epoch}")

# 使用
callbacks = [
    MyCallbackRegistry.build("LoggingCallback"),
    MyCallbackRegistry.build("CheckpointCallback"),
]
```

### 7.2 注册表配置选项

```python
# 允许覆盖已注册组件
FlexibleRegistry = Registry("flexible", allow_override=True)

# 带基类验证
from torch.utils.data import Dataset
DatasetRegistry = Registry("datasets", base_class=Dataset)
```

---

## 8. 最佳实践

### 8.1 命名规范

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Backbone | `{Model}Backbone` | `Qwen2VLBackbone` |
| Pooling | `{Method}Pooling` | `LastTokenPooling`, `MeanPooling` |
| Loss | `{Method}Loss` | `InfoNCELoss`, `ContrastiveLoss` |
| Dataset | `{Source}Dataset` | `MMEBDataset`, `ViDoReDataset` |

### 8.2 注册时机

```python
# ✅ 推荐：在模块定义文件中注册
# src/vlm2emb/modules/pooling.py

from vlm2emb import AutoModule

@AutoModule.register("LastTokenPooling")
class LastTokenPooling:
    ...

@AutoModule.register("MeanPooling")
class MeanPooling:
    ...
```

```python
# ✅ 确保模块被导入
# src/vlm2emb/modules/__init__.py

from vlm2emb.modules.pooling import LastTokenPooling, MeanPooling

__all__ = ["LastTokenPooling", "MeanPooling"]
```

### 8.3 避免循环导入

```python
# ❌ 避免：在 __init__.py 中导入大量模块
# src/vlm2emb/__init__.py
from vlm2emb.modules import *  # 可能导致循环导入

# ✅ 推荐：延迟导入或显式导入
# src/vlm2emb/__init__.py
from vlm2emb.auto import AutoModule, AutoModel  # 只导入注册表
```

---

## 9. 异常处理

### 9.1 ComponentNotFoundError

```python
from vlm2emb import AutoModule
from vlm2emb.exceptions import ComponentNotFoundError

try:
    module = AutoModule.build("NonExistentModule")
except ComponentNotFoundError as e:
    print(f"组件未找到: {e}")
    print(f"可用组件: {AutoModule.list_modules()}")
```

### 9.2 ComponentAlreadyExistsError

```python
from vlm2emb import AutoModule
from vlm2emb.exceptions import ComponentAlreadyExistsError

@AutoModule.register("MyModule")
class MyModule:
    pass

try:
    @AutoModule.register("MyModule")  # 重复注册
    class AnotherModule:
        pass
except ComponentAlreadyExistsError:
    print("组件已存在，使用 override=True 覆盖")

# 使用 override=True 覆盖
@AutoModule.register("MyModule", override=True)
class UpdatedModule:
    pass
```

---

## 10. 相关文档

- [架构总览](./overview.md) - 整体架构设计
- [模块管道系统](./module-pipeline.md) - AutoModule 使用场景
- [配置系统](./config-system.md) - type 字段与注册表集成
- [BToks API](../api/model.md) - 模型 API 参考
