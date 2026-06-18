# BToks API Reference

> **Version**: 0.1
> **Updated**: 2026-01-25
> **Module**: `vlm2emb.model`

This document provides detailed reference for BToks core API.

---

## Table of Contents

1. [create_model](#create_model) - Model factory function
2. [BToksConfig](#btoksconfig) - Primary BToks model configuration class
3. [BToks](#btoks) - Core model class

---

## create_model

Unified entry function for creating models from configuration.

### Function Signature

```python
def create_model(config: Config | dict) -> PreTrainedModel:
    """Create model from configuration (unified entry point).
    """
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | `Config \| dict` | ✅ | Loaded configuration (OmegaConf Config or dict) |

### Return Value

| Type | Description |
|------|-------------|
| `PreTrainedModel` | Model instance (usually `BToks`) |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Model type not registered to HuggingFace AutoConfig |

### Examples

```python
from vlm2emb import create_model
from vlm2emb.config import load_config

# Method 1: Create from YAML config (recommended for training)
config = load_config("configs/presets/btoks_qwen2vl_2b_v1.yaml")
model = create_model(config.model)

# Method 2: Create from dict
model_config = {
    "type": "BToks",
    "modules": {
        "injector": {"type": "BToksTokenInjector", "num_tokens": 4},
        "backbone": {"type": "Qwen2VLBackbone", "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct"},
        "pooling": {"type": "BToksPooling"},
        "normalize": {"type": "Normalize"},
    },
}
model = create_model(model_config)
```

### Implementation Notes

`create_model` internally performs the following steps:

1. Convert OmegaConf configuration to dict (if needed)
2. Read the model configuration section passed by the caller
3. Get `type` field (default `"VLM2Emb"` for the compatibility model)
4. Use HuggingFace `AutoConfig.for_model()` to create config
5. Use HuggingFace `AutoModel.from_config()` to create model
6. **Post-build step**: Iterate modules and attach `backbone_config` (serialized structure config) to Backbone modules, producing a dual-field runtime config:
   - `model_name_or_path`: weight source for fresh build / reference restore
   - `backbone_config`: structure snapshot for artifact restore

---

## BToksConfig

Primary BToks model configuration class. `VLM2EmbConfig` remains available as a
compatibility name for the Python package namespace.

### Class Definition

```python
class BToksConfig(VLM2EmbConfig):
    """Configuration class for BToks model."""

    model_type = "btoks"
```

## VLM2EmbConfig

Compatibility model configuration class, inherits from `transformers.PretrainedConfig`.

### Class Definition

```python
class VLM2EmbConfig(PretrainedConfig):
    """Configuration class for BToks model.

    BToks 模型的配置类。
    """

    model_type = "vlm2emb"
```

### Constructor

```python
def __init__(
    self,
    modules: list[dict[str, Any]] | None = None,
    **kwargs,
):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modules` | `list[dict]` | `[]` | Module configuration list. Each dict must have `type` key |
| `**kwargs` | - | - | Extra parameters passed to `PretrainedConfig` |

> **Note**: Following HuggingFace standard, `config.json` does not include Processor configuration. Processor needs to be saved and loaded separately.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_type` | `str` | Fixed as `"vlm2emb"` |
| `modules` | `list[dict]` | Module configuration list |

### Examples

```python
from vlm2emb import VLM2EmbConfig

# Create VLM2Vec-style configuration (without processor)
config = VLM2EmbConfig(
    modules=[
        {
            "type": "Qwen2VLBackbone",
            "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },
        {"type": "LastTokenPooling"},
        {"type": "Normalize"},
    ],
)

# Save config
config.save_pretrained("./my_model")

# Load config
config = VLM2EmbConfig.from_pretrained("./my_model")
```

### modules Configuration Format

Each module configuration dict must contain `type` key, other keys are module-specific parameters:

```python
modules = [
    # Backbone module
    {
        "type": "Qwen2VLBackbone",           # Required: module type
        "model_name_or_path": "...",         # Backbone-specific parameters
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "min_pixels": 200704,
        "max_pixels": 1003520,
    },
    # Pooling module
    {
        "type": "LastTokenPooling",          # No extra parameters
    },
    # Or MeanPooling
    {
        "type": "MeanPooling",
    },
    # Normalize module
    {
        "type": "Normalize",
    },
]
```

---

## BToks

Core model class, inherits from `transformers.PreTrainedModel`.

### Class Definition

```python
class BToks(PreTrainedModel):
    """BToks: Vision-Language Model to Embedding Model.

    BToks：视觉-语言模型到嵌入模型。
    """

    config_class = VLM2EmbConfig
    base_model_prefix = "vlm2emb"
    supports_gradient_checkpointing = True
```

### Constructor

```python
def __init__(self, config: VLM2EmbConfig):
    """Initialize BToks model.

    初始化 BToks 模型。

    Args:
        config: Model configuration with modules list
               包含 modules 列表的模型配置
    """
```

---

### forward

Training forward pass method.

#### Signature

```python
def forward(
    self,
    input_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    pixel_values: Tensor | None = None,
    image_grid_thw: Tensor | None = None,
    pixel_values_videos: Tensor | None = None,
    video_grid_thw: Tensor | None = None,
    **kwargs,
) -> dict[str, Tensor]:
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_ids` | `Tensor` | ✅ | Token IDs, shape `(B, L)` |
| `attention_mask` | `Tensor` | ❌ | Attention mask, shape `(B, L)` |
| `pixel_values` | `Tensor` | ❌ | Image pixel values |
| `image_grid_thw` | `Tensor` | ❌ | Image grid dimensions (Qwen2VL) |
| `pixel_values_videos` | `Tensor` | ❌ | Video pixel values |
| `video_grid_thw` | `Tensor` | ❌ | Video grid dimensions |
| `**kwargs` | - | ❌ | Extra parameters (passed to modules) |

#### Return Value

| Type | Description |
|------|-------------|
| `dict[str, Tensor]` | Features dict, at least containing `embeddings` key |

**Standard keys in returned dict**:

| Key | Shape | Description |
|-----|-------|-------------|
| `embeddings` | `(B, D)` | Final embedding vectors |
| `last_hidden_state` | `(B, L, D)` | Backbone output |
| `attention_mask` | `(B, L)` | Attention mask |

#### Examples

```python
# Training usage
model.train()
features = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    pixel_values=batch["pixel_values"],
    image_grid_thw=batch["image_grid_thw"],
)

embeddings = features["embeddings"]  # (B, D)
loss = loss_fn(query_emb, positive_emb)
```

---

### Other Methods

#### device

Get model device.

```python
@property
def device(self) -> torch.device:
    """Get model device."""
```

**Example**:
```python
print(model.device)  # cuda:0 or cpu
```

#### get_module

Get module in pipeline by index.

```python
def get_module(self, index: int) -> nn.Module:
    """Get module by index.

    通过索引获取模块。
    """
```

**Example**:
```python
backbone = model.get_module(0)  # Get Backbone
pooling = model.get_module(1)   # Get Pooling
```

#### \_\_len\_\_

Return number of modules in pipeline.

```python
def __len__(self) -> int:
    """Return number of modules in pipeline."""
```

**Example**:
```python
print(len(model))  # 3 (Backbone + Pooling + Normalize)
```

#### \_\_iter\_\_

Iterate over modules in pipeline.

```python
def __iter__(self):
    """Iterate over modules."""
```

**Example**:
```python
for module in model:
    print(type(module).__name__)
# Output:
# Qwen2VLBackbone
# LastTokenPooling
# Normalize
```

---

### Class Methods (Inherited from PreTrainedModel)

#### from_pretrained

Load model from pretrained weights.

```python
model = BToks.from_pretrained(
    "public-model-or-checkpoint",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

#### save_pretrained

Save model to directory.

```python
model.save_pretrained("./my_model")
```

> **Note**: Training recipe config and saved artifact config are different schemas.
>
> - Recipe config usually only stores `model_name_or_path` for fresh builds
> - Reference artifacts keep both:
>   - `backbone_config` for skeleton construction
>   - `model_name_or_path` for a single post-load backbone materialization
> - Full artifacts keep at least `backbone_config`

#### from_pretrained Loading Behavior

When loading via `from_pretrained`, Backbone first initializes through `backbone_config` in structure mode (creates model structure only). Then:

- Full artifacts restore backbone weights entirely from the local checkpoint
- Reference artifacts restore custom/base weights locally, then materialize backbone once from `model_name_or_path`

```python
# Example backbone config in saved config.json:
# {
#   "modules": {
#     "backbone": {
#       "type": "Qwen2VLBackbone",
#       "backbone_config": { ... },            # Structure snapshot
#       "model_name_or_path": "Qwen/...",      # Reference restore source
#       "dtype": "bfloat16"
#     }
#   }
# }
```

#### push_to_hub

Upload model to HuggingFace Hub.

```python
model.push_to_hub("public-model-or-checkpoint")
```

---

## Complete Examples

### Inference Flow

```python
import torch
from vlm2emb import BToks
from transformers import AutoProcessor
from PIL import Image

# 1. Load model and processor (separate loading, follows HF standard)
model = BToks.from_pretrained("public-model-or-checkpoint")
processor = AutoProcessor.from_pretrained("public-model-or-checkpoint")
model.eval()

# 2. Prepare data
image = Image.open("test.jpg")
texts = ["What is shown in this image?", "A photo of a cat"]

# 3. Preprocess
inputs = processor(text=texts, images=[image, image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 4. Forward pass to get embeddings
with torch.no_grad():
    features = model(**inputs)

embeddings = features["embeddings"]  # (B, D)

# 5. Subsequent processing (similarity calculation, etc.) implemented by user
print(f"Embeddings shape: {embeddings.shape}")
```

### Training Flow

```python
from vlm2emb import create_model
from vlm2emb.config import load_config
from vlm2emb.training import Trainer
from transformers import TrainingArguments

# 1. Load config and create model
config = load_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
model = create_model(config)

# 2. Set training arguments
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
)

# 3. Create Trainer and train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collator,
)

trainer.train()

# 4. Save model
model.save_pretrained("./trained_model")
```

---

## Related Documents

- [Architecture Overview](../architecture/overview.md) - Overall architecture design
- [Module Pipeline System](../architecture/module-pipeline.md) - Module design details
- [Configuration System](../architecture/config-system.md) - YAML configuration details
- [Trainer API](./trainer.md) - Trainer API Reference
