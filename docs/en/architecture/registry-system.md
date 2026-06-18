# Registry System

> **Version**: 0.1
> **Updated**: 2026-01-25
> **Architecture Design Version**: 1.0
> **Reference**: HuggingFace transformers Auto Classes

## 1. Design Concept

### 1.1 Why Registry is Needed

BToks uses registry pattern to manage all pluggable components:

| Goal | Description |
|------|-------------|
| **Configuration-Driven** | Specify components via `type` field in YAML, no hardcoding |
| **Dynamic Discovery** | Automatically discover registered components at runtime |
| **Easy Extension** | New components only need decorator registration, no framework code changes |
| **Type Safety** | Optional base class validation, ensuring component interface consistency |

### 1.2 Comparison with HuggingFace Auto Classes

BToks's registry design references HuggingFace transformers Auto Classes:

```python
# HuggingFace style
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# BToks style
from vlm2emb import AutoModule
backbone = AutoModule.from_config({"type": "Qwen2VL", ...})
```

**Main Differences**:

| Aspect | HuggingFace | BToks |
|--------|-------------|---------|
| Component Discovery | Based on model_type mapping | Based on explicit registration |
| Instantiation | `from_pretrained()` | `from_config()` / `build()` |
| Registration | Internal mapping table | Decorator / direct registration |

---

## 2. Registry Base Class

### 2.1 Class Definition

```python
from vlm2emb import Registry

class Registry:
    """Generic component registry.

    Attributes:
        name: Registry name (e.g., "backbone", "dataset")
        base_class: Optional base class for validating registered components
        _components: Internal mapping of name to component
    """

    def __init__(
        self,
        name: str,
        base_class: type | None = None,
        allow_override: bool = False,
    ):
        ...
```

### 2.2 Core Methods

#### register() - Register Component

```python
def register(
    self,
    name: str | None = None,
    component: type | Callable | None = None,
    override: bool = False,
) -> type | Callable:
    """Register component class or factory function.

    Supports two modes:
    1. Decorator: @Registry.register("name")
    2. Direct call: Registry.register("name", Component)
    """
```

**Usage Examples**:

```python
from vlm2emb import AutoModule

# Mode 1: Decorator + specified name
@AutoModule.register("MyPooling")
class MyPoolingModule:
    pass

# Mode 2: Decorator + auto-use class name
@AutoModule.register()
class WeightedPooling:
    pass  # Registered as "WeightedPooling"

# Mode 3: Direct registration
AutoModule.register("CustomModule", CustomModuleClass)
```

#### from_config() - Instantiate from Config

```python
def from_config(self, config: dict, **kwargs) -> Any:
    """Instantiate component from config dict.

    Args:
        config: Config dict must contain 'type' key
        **kwargs: Extra parameters to override config

    Returns:
        Instantiated component
    """
```

**Usage Examples**:

```python
from vlm2emb import AutoModule

# Create from config
backbone = AutoModule.from_config({
    "type": "Qwen2VLBackbone",
    "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
    "dtype": "bfloat16",
})

# Override config parameters
backbone = AutoModule.from_config(
    {"type": "Qwen2VLBackbone", "model_name_or_path": "..."},
    dtype="float16",  # Override dtype
)
```

#### build() - Direct Build

```python
def build(self, name: str, *args, **kwargs) -> Any:
    """Build component instance directly by name."""
```

**Usage Examples**:

```python
from vlm2emb import AutoModule

# Direct build
pooling = AutoModule.build("LastTokenPooling")
normalize = AutoModule.build("Normalize")
```

#### Other Methods

| Method | Description |
|--------|-------------|
| `get(name)` | Get registered component class (not instantiated) |
| `list_modules()` | List all registered component names |
| `is_registered(name)` | Check if component is registered |
| `__contains__(name)` | Support `in` operator |
| `__len__()` | Return number of registered components |

---

## 3. Auto* Predefined Registries

BToks provides the following predefined registries:

### 3.1 Model-Related

| Registry | Description | Typical Components |
|----------|-------------|-------------------|
| `AutoModel` | Complete models | BToks |
| `AutoModule` | Pipeline modules | Qwen2VLBackbone, LastTokenPooling, Normalize |

### 3.2 Training-Related

| Registry | Description | Typical Components |
|----------|-------------|-------------------|
| `AutoTrainer` | Trainers | VLM2VecTrainer |
| `AutoLoss` | Loss functions | InfoNCELoss, ContrastiveLoss |
| `AutoOptimizer` | Optimizers | AdamW |
| `AutoScheduler` | LR schedulers | CosineScheduler |

### 3.3 Data-Related

| Registry | Description | Typical Components |
|----------|-------------|-------------------|
| `AutoDataset` | Datasets | MMEBDataset, ViDoReDataset |
| `AutoCollator` | Data collators | TrainingCollator |
| `AutoSampler` | Data samplers | DistributedSampler |

### 3.4 Evaluation-Related

| Registry | Description | Typical Components |
|----------|-------------|-------------------|
| `AutoMetric` | Evaluation metrics | Recall, NDCG |
| `AutoEvaluator` | Evaluators | MMEBEvaluator |

### 3.5 Usage Examples

```python
from vlm2emb import (
    AutoModule,
    AutoLoss,
    AutoDataset,
)

# List available components
print(AutoModule.list_modules())
# ['Qwen2VLBackbone', 'LastTokenPooling', 'MeanPooling', 'Normalize', ...]

# Check if component exists
if "Qwen2VLBackbone" in AutoModule:
    backbone = AutoModule.build("Qwen2VLBackbone", ...)
```

---

## 4. Component Registration

### 4.1 Decorator Registration (Recommended)

Most common registration method, automatically registers at class definition:

```python
from vlm2emb import AutoModule
import torch.nn as nn

@AutoModule.register("WeightedPooling")
class WeightedPooling(nn.Module):
    """Custom weighted pooling module."""

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

### 4.2 Factory Function Registration

For components requiring complex initialization logic, register factory functions:

```python
from vlm2emb import AutoDataset

@AutoDataset.register("custom_dataset")
def load_custom_dataset(data_path: str, split: str = "train", **kwargs):
    """Factory function: load custom dataset."""
    from datasets import load_dataset
    return load_dataset(data_path, split=split, **kwargs)
```

### 4.3 Direct Registration

Suitable for registering third-party classes or dynamic registration:

```python
from vlm2emb import AutoLoss
from some_library import CustomLoss

# Register third-party class directly
AutoLoss.register("CustomLoss", CustomLoss)

# Dynamic registration
def register_losses():
    for name, cls in discover_loss_classes():
        AutoLoss.register(name, cls)
```

### 4.4 Registration with Base Class Validation

Create custom registry with base class validation:

```python
from vlm2emb import Registry
import torch.nn as nn

# Create registry requiring nn.Module inheritance
ModuleRegistry = Registry("custom_modules", base_class=nn.Module)

@ModuleRegistry.register("ValidModule")
class ValidModule(nn.Module):  # ✅ Passes validation
    pass

@ModuleRegistry.register("InvalidModule")
class InvalidModule:  # ❌ TypeError: Must inherit nn.Module
    pass
```

---

## 5. Configuration Integration

### 5.1 type Field in YAML Configuration

Registries are tightly integrated with configuration system, specify components via `type` field:

```yaml
# configs/models/vlm2vec.yaml
model:
  type: vlm2emb  # AutoModel lookup

  modules:
    - type: Qwen2VLBackbone  # AutoModule lookup
      model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
      dtype: bfloat16

    - type: LastTokenPooling  # AutoModule lookup

    - type: Normalize  # AutoModule lookup

training:
  loss:
    type: InfoNCELoss  # AutoLoss lookup
    temperature: 0.02
```

### 5.2 from_config and from_config Class Methods

If component class defines `from_config` class method, Registry prioritizes it:

```python
@AutoModule.register("ComplexModule")
class ComplexModule(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        ...

    @classmethod
    def from_config(cls, config: dict) -> "ComplexModule":
        """Custom config parsing logic."""
        config = config.copy()
        config.pop("type", None)

        # Custom logic: set parameters based on model_size
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
# Use custom from_config
modules:
  - type: ComplexModule
    model_size: large  # from_config converts to hidden_size=1024, num_layers=12
```

---

## 6. Utility Functions

### 6.1 get_registry()

Get predefined registry by name:

```python
from vlm2emb import get_registry

backbone_registry = get_registry("backbone")
print(backbone_registry.list_modules())
```

### 6.2 list_registries()

List all available registries:

```python
from vlm2emb import list_registries

print(list_registries())
# ['backbone', 'head', 'model', 'module', 'trainer', 'loss', ...]
```

---

## 7. Custom Registries

### 7.1 Creating Custom Registry

```python
from vlm2emb import Registry

# Create custom registry
MyCallbackRegistry = Registry("callback")

@MyCallbackRegistry.register("LoggingCallback")
class LoggingCallback:
    def on_step_end(self, step, loss):
        print(f"Step {step}: loss={loss:.4f}")

@MyCallbackRegistry.register("CheckpointCallback")
class CheckpointCallback:
    def on_epoch_end(self, epoch, model):
        model.save_pretrained(f"checkpoint-{epoch}")

# Usage
callbacks = [
    MyCallbackRegistry.build("LoggingCallback"),
    MyCallbackRegistry.build("CheckpointCallback"),
]
```

### 7.2 Registry Configuration Options

```python
# Allow overriding registered components
FlexibleRegistry = Registry("flexible", allow_override=True)

# With base class validation
from torch.utils.data import Dataset
DatasetRegistry = Registry("datasets", base_class=Dataset)
```

---

## 8. Best Practices

### 8.1 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Backbone | `{Model}Backbone` | `Qwen2VLBackbone` |
| Pooling | `{Method}Pooling` | `LastTokenPooling`, `MeanPooling` |
| Loss | `{Method}Loss` | `InfoNCELoss`, `ContrastiveLoss` |
| Dataset | `{Source}Dataset` | `MMEBDataset`, `ViDoReDataset` |

### 8.2 Registration Timing

```python
# ✅ Recommended: Register in module definition file
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
# ✅ Ensure modules are imported
# src/vlm2emb/modules/__init__.py

from vlm2emb.modules.pooling import LastTokenPooling, MeanPooling

__all__ = ["LastTokenPooling", "MeanPooling"]
```

### 8.3 Avoiding Circular Imports

```python
# ❌ Avoid: Importing many modules in __init__.py
# src/vlm2emb/__init__.py
from vlm2emb.modules import *  # May cause circular imports

# ✅ Recommended: Lazy import or explicit import
# src/vlm2emb/__init__.py
from vlm2emb.auto import AutoModule, AutoModel  # Only import registries
```

---

## 9. Exception Handling

### 9.1 ComponentNotFoundError

```python
from vlm2emb import AutoModule
from vlm2emb.exceptions import ComponentNotFoundError

try:
    module = AutoModule.build("NonExistentModule")
except ComponentNotFoundError as e:
    print(f"Component not found: {e}")
    print(f"Available components: {AutoModule.list_modules()}")
```

### 9.2 ComponentAlreadyExistsError

```python
from vlm2emb import AutoModule
from vlm2emb.exceptions import ComponentAlreadyExistsError

@AutoModule.register("MyModule")
class MyModule:
    pass

try:
    @AutoModule.register("MyModule")  # Duplicate registration
    class AnotherModule:
        pass
except ComponentAlreadyExistsError:
    print("Component exists, use override=True to override")

# Use override=True to override
@AutoModule.register("MyModule", override=True)
class UpdatedModule:
    pass
```

---

## 10. Related Documents

- [Architecture Overview](./overview.md) - Overall architecture design
- [Module Pipeline System](./module-pipeline.md) - AutoModule usage scenarios
- [Configuration System](./config-system.md) - type field and registry integration
- [BToks API](../api/model.md) - Model API Reference
