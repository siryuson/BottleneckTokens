# BToks API 参考

> **版本**: 0.1
> **更新日期**: 2026-01-25
> **模块**: `vlm2emb.model`

本文档提供 BToks 核心 API 的详细参考。

---

## 目录

1. [create_model](#create_model) - 模型工厂函数
2. [BToksConfig](#btoksconfig) - 主要 BToks 模型配置类
3. [BToks](#btoks) - 核心模型类

---

## create_model

从配置创建模型的统一入口函数。

### 函数签名

```python
def create_model(config: Config | dict) -> PreTrainedModel:
    """Create model from configuration (unified entry point).

    从配置创建模型（统一入口）。
    """
```

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `config` | `Config \| dict` | ✅ | 已加载的配置（OmegaConf Config 或 dict） |

### 返回值

| 类型 | 说明 |
|------|------|
| `PreTrainedModel` | 模型实例（通常是 `BToks`） |

### 异常

| 异常 | 条件 |
|------|------|
| `ValueError` | 模型类型未注册到 HuggingFace AutoConfig |

### 示例

```python
from vlm2emb import create_model
from vlm2emb.config import load_config

# 方式 1: 从 YAML 配置创建（推荐用于训练）
config = load_config("configs/presets/btoks_qwen2vl_2b_v1.yaml")
model = create_model(config.model)

# 方式 2: 从字典创建
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

### 实现说明

`create_model` 内部执行以下步骤：

1. 将 OmegaConf 配置转换为 dict（如果需要）
2. 读取调用方传入的 model 配置部分
3. 获取 `type` 字段（兼容模型默认 `"VLM2Emb"`）
4. 使用 HuggingFace `AutoConfig.for_model()` 创建配置
5. 使用 HuggingFace `AutoModel.from_config()` 创建模型
6. **Post-build 步骤**：遍历 modules，为 Backbone 补充 `backbone_config`（序列化后的结构配置），形成运行时双字段配置：
   - `model_name_or_path`：fresh build / reference 恢复时的权重来源
   - `backbone_config`：artifact 恢复时的结构快照

---

## BToksConfig

主要 BToks 模型配置类。`VLM2EmbConfig` 作为 Python package namespace 的兼容名称保留。

### 类定义

```python
class BToksConfig(VLM2EmbConfig):
    """Configuration class for BToks model."""

    model_type = "btoks"
```

## VLM2EmbConfig

兼容模型配置类，继承自 `transformers.PretrainedConfig`。

### 类定义

```python
class VLM2EmbConfig(PretrainedConfig):
    """Configuration class for BToks model.

    BToks 模型的配置类。
    """

    model_type = "vlm2emb"
```

### 构造函数

```python
def __init__(
    self,
    modules: list[dict[str, Any]] | None = None,
    **kwargs,
):
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `modules` | `list[dict]` | `[]` | 模块配置列表。每个字典必须有 `type` 键 |
| `**kwargs` | - | - | 传递给 `PretrainedConfig` 的额外参数 |

> **注意**：遵循 HuggingFace 标准，`config.json` 不包含 Processor 配置。Processor 需单独保存和加载。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `model_type` | `str` | 固定为 `"vlm2emb"` |
| `modules` | `list[dict]` | 模块配置列表 |

### 示例

```python
from vlm2emb import VLM2EmbConfig

# 创建 VLM2Vec 风格配置（不含 processor）
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

# 保存配置
config.save_pretrained("./my_model")

# 加载配置
config = VLM2EmbConfig.from_pretrained("./my_model")
```

### modules 配置格式

每个模块配置字典必须包含 `type` 键，其他键为模块特定参数：

```python
modules = [
    # Backbone 模块
    {
        "type": "Qwen2VLBackbone",           # 必需：模块类型
        "model_name_or_path": "...",         # Backbone 特定参数
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "min_pixels": 200704,
        "max_pixels": 1003520,
    },
    # Pooling 模块
    {
        "type": "LastTokenPooling",          # 无额外参数
    },
    # 或 MeanPooling
    {
        "type": "MeanPooling",
    },
    # Normalize 模块
    {
        "type": "Normalize",
    },
]
```

---

## BToks

核心模型类，继承自 `transformers.PreTrainedModel`。

### 类定义

```python
class BToks(PreTrainedModel):
    """BToks: Vision-Language Model to Embedding Model.

    BToks：视觉-语言模型到嵌入模型。
    """

    config_class = VLM2EmbConfig
    base_model_prefix = "vlm2emb"
    supports_gradient_checkpointing = True
```

### 构造函数

```python
def __init__(self, config: VLM2EmbConfig):
    """Initialize BToks model.

    初始化 BToks 模型。

    Args:
        config: 包含 modules 列表的模型配置
               Model configuration with modules list
    """
```

---

### forward

训练时的前向传播方法。

#### 签名

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

#### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `input_ids` | `Tensor` | ✅ | Token IDs，形状 `(B, L)` |
| `attention_mask` | `Tensor` | ❌ | 注意力掩码，形状 `(B, L)` |
| `pixel_values` | `Tensor` | ❌ | 图像像素值 |
| `image_grid_thw` | `Tensor` | ❌ | 图像网格维度（Qwen2VL） |
| `pixel_values_videos` | `Tensor` | ❌ | 视频像素值 |
| `video_grid_thw` | `Tensor` | ❌ | 视频网格维度 |
| `**kwargs` | - | ❌ | 额外参数（传递给模块） |

#### 返回值

| 类型 | 说明 |
|------|------|
| `dict[str, Tensor]` | Features 字典，至少包含 `embeddings` 键 |

**返回字典的标准键**：

| 键 | 形状 | 说明 |
|-----|------|------|
| `embeddings` | `(B, D)` | 最终嵌入向量 |
| `last_hidden_state` | `(B, L, D)` | Backbone 输出 |
| `attention_mask` | `(B, L)` | 注意力掩码 |

#### 示例

```python
# 训练时使用
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

### 其他方法

#### device

获取模型所在设备。

```python
@property
def device(self) -> torch.device:
    """Get model device."""
```

**示例**：
```python
print(model.device)  # cuda:0 或 cpu
```

#### get_module

通过索引获取管道中的模块。

```python
def get_module(self, index: int) -> nn.Module:
    """Get module by index.

    通过索引获取模块。
    """
```

**示例**：
```python
backbone = model.get_module(0)  # 获取 Backbone
pooling = model.get_module(1)   # 获取 Pooling
```

#### \_\_len\_\_

返回管道中的模块数量。

```python
def __len__(self) -> int:
    """Return number of modules in pipeline."""
```

**示例**：
```python
print(len(model))  # 3 (Backbone + Pooling + Normalize)
```

#### \_\_iter\_\_

迭代管道中的模块。

```python
def __iter__(self):
    """Iterate over modules."""
```

**示例**：
```python
for module in model:
    print(type(module).__name__)
# 输出:
# Qwen2VLBackbone
# LastTokenPooling
# Normalize
```

---

### 类方法（继承自 PreTrainedModel）

#### from_pretrained

从预训练权重加载模型。

```python
model = BToks.from_pretrained(
    "public-model-or-checkpoint",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

#### save_pretrained

保存模型到目录。

```python
model.save_pretrained("./my_model")
```

> **注意**：训练配置（recipe config）和保存产物（artifact config）不是同一种 schema。
>
> - recipe config 通常只含 `model_name_or_path`，用于 fresh build
> - artifact config 在 reference 模式下同时保留：
>   - `backbone_config`：用于 skeleton 构建
>   - `model_name_or_path`：用于单次后置重载 reference backbone
> - artifact config 在 full 模式下至少保留 `backbone_config`

#### from_pretrained 加载行为

`from_pretrained` 加载时，Backbone 先通过 `backbone_config` 走结构模式初始化（仅创建模型结构）。随后：

- full artifact：由本地 checkpoint 权重完整恢复，不访问外部 `model_name_or_path`
- reference artifact：使用本地 custom/base 权重恢复其它模块，再根据 `model_name_or_path` 单次重载 backbone

```python
# 保存后的 config.json 中 backbone 配置示例：
# {
#   "modules": {
#     "backbone": {
#       "type": "Qwen2VLBackbone",
#       "backbone_config": { ... },            # 结构配置
#       "model_name_or_path": "Qwen/...",      # reference 恢复来源
#       "dtype": "bfloat16"
#     }
#   }
# }
```

#### push_to_hub

上传模型到 HuggingFace Hub。

```python
model.push_to_hub("public-model-or-checkpoint")
```

---

## 完整示例

### 推理流程

```python
import torch
from vlm2emb import BToks
from transformers import AutoProcessor
from PIL import Image

# 1. 加载模型和 processor（分离加载，遵循 HF 标准）
model = BToks.from_pretrained("public-model-or-checkpoint")
processor = AutoProcessor.from_pretrained("public-model-or-checkpoint")
model.eval()

# 2. 准备数据
image = Image.open("test.jpg")
texts = ["What is shown in this image?", "A photo of a cat"]

# 3. 预处理
inputs = processor(text=texts, images=[image, image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 4. 前向传播获取嵌入
with torch.no_grad():
    features = model(**inputs)

embeddings = features["embeddings"]  # (B, D)

# 5. 后续处理（如相似度计算）由用户自行实现
print(f"Embeddings shape: {embeddings.shape}")
```

### 训练流程

```python
from vlm2emb import create_model
from vlm2emb.config import load_config
from vlm2emb.training import Trainer
from transformers import TrainingArguments

# 1. 加载配置和创建模型
config = load_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
model = create_model(config)

# 2. 设置训练参数
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
)

# 3. 创建 Trainer 并训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collator,
)

trainer.train()

# 4. 保存模型
model.save_pretrained("./trained_model")
```

---

## 相关文档

- [架构总览](../architecture/overview.md) - 整体架构设计
- [模块管道系统](../architecture/module-pipeline.md) - 模块设计详解
- [配置系统](../architecture/config-system.md) - YAML 配置详解
- [Trainer API](./trainer.md) - 训练器 API 参考
