# Trainer API 参考

> **版本**: 0.1
> **更新日期**: 2026-01-25
> **模块**: `vlm2emb.training`

本文档提供 BToks 训练器 API 的详细参考。

---

## 目录

1. [Trainer](#trainer) - 基础嵌入训练器
2. [VLM2VecTrainer](#vlm2vectrainer) - VLM2Vec 训练器
3. [VLM2VecTrainingArgs](#vlm2vectrainingargs) - VLM2Vec 训练参数
4. [create_trainer](#create_trainer) - 训练器工厂函数

---

## Trainer

基础嵌入模型训练器，继承自 HuggingFace Trainer。

### 类定义

```python
from vlm2emb.training import Trainer

class Trainer(transformers.Trainer):
    """Generic trainer for embedding models.

    嵌入模型通用训练器。
    """
```

### 构造函数

```python
def __init__(
    self,
    model: PreTrainedModel | nn.Module | None = None,
    args: TrainingArguments | None = None,
    data_collator: Callable | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    processing_class: Any | None = None,
    model_init: Callable[[], PreTrainedModel] | None = None,
    compute_metrics: Callable | None = None,
    callbacks: list | None = None,
    optimizers: tuple = (None, None),
    preprocess_logits_for_metrics: Callable | None = None,
    **kwargs,
):
```

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | `PreTrainedModel \| nn.Module` | ❌ | 要训练的模型 |
| `args` | `TrainingArguments` | ❌ | 训练参数 |
| `data_collator` | `Callable` | ❌ | 数据整理器 |
| `train_dataset` | `Dataset` | ❌ | 训练数据集 |
| `eval_dataset` | `Dataset` | ❌ | 评估数据集 |
| `processing_class` | `Any` | ❌ | Processor/Tokenizer |
| `model_init` | `Callable` | ❌ | 模型初始化函数 |
| `compute_metrics` | `Callable` | ❌ | 指标计算函数 |
| `callbacks` | `list` | ❌ | 训练回调列表 |
| `optimizers` | `tuple` | ❌ | (optimizer, scheduler) 元组 |
| `preprocess_logits_for_metrics` | `Callable` | ❌ | logits 预处理函数 |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `loss_fn` | `nn.Module` | 对比损失函数（自动根据分布式环境选择） |
| `accelerator` | `Accelerator` | HuggingFace Accelerator 实例 |

### 方法

#### compute_loss()

计算对比损失。

```python
def compute_loss(
    self,
    model: nn.Module,
    inputs: dict[str, dict],
    return_outputs: bool = False,
    num_items_in_batch: int | None = None,
) -> Tensor | tuple[Tensor, Any]:
    """Compute contrastive loss for (qry, pos) pairs.

    计算 (qry, pos) 对的对比损失。

    Args:
        model: 模型
        inputs: 包含 "query", "positive", "metadata" 键的字典
        return_outputs: 是否返回模型输出
        num_items_in_batch: batch 中的样本数（未使用）

    Returns:
        损失张量，可选地包含输出
    """
```

**输入格式**：

```python
inputs = {
    "query": {
        "input_ids": Tensor,        # (B, L_q)
        "attention_mask": Tensor,   # (B, L_q)
        "pixel_values": Tensor,     # (N_q, C, H, W)
        "image_grid_thw": Tensor,   # (N_q, 3)
    },
    "positive": {
        "input_ids": Tensor,        # (B, L_p)
        "attention_mask": Tensor,   # (B, L_p)
        "pixel_values": Tensor,     # (N_p, C, H, W)
        "image_grid_thw": Tensor,   # (N_p, 3)
    },
    "metadata": [...]               # 可选
}
```

**示例**：

```python
# 通常由 Trainer 内部调用
loss = trainer.compute_loss(model, batch)

# 返回输出
loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
qry_embeds = outputs["qry_embeds"]
pos_embeds = outputs["pos_embeds"]
```

#### _setup_loss_fn()

设置对比损失函数。

```python
def _setup_loss_fn(self) -> None:
    """Setup contrastive loss function.

    设置对比损失函数。
    根据分布式环境自动选择 ContrastiveLoss 或 DistributedContrastiveLoss。
    """
```

**行为**：
- 分布式环境：使用 `DistributedContrastiveLoss`
- 单机环境：使用 `ContrastiveLoss`

#### _get_train_sampler()

获取训练采样器。

```python
def _get_train_sampler(
    self,
    train_dataset: Dataset | None = None
) -> Sampler | None:
    """Get training sampler with ChunkSampler support.

    获取训练采样器，支持 ChunkSampler。

    Args:
        train_dataset: 训练数据集

    Returns:
        Sampler 实例或 None
    """
```

**行为**：
- 如果 `args.interleave_batch_size > 0`：返回 `ChunkSampler`
- 否则：返回默认 `RandomSampler`

### 示例

```python
from vlm2emb.training import Trainer
from transformers import TrainingArguments

# 创建训练参数
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
    save_strategy="steps",
    save_steps=500,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

# 训练
trainer.train()

# 从检查点恢复
trainer.train(resume_from_checkpoint="./output/checkpoint-500")
```

---

## VLM2VecTrainer

VLM2Vec 专用训练器，支持 GradCache。

### 类定义

```python
from vlm2emb.training.trainers import VLM2VecTrainer

class VLM2VecTrainer(Trainer):
    """Trainer for VLM2Vec embedding model with optional GradCache.

    VLM2Vec 嵌入模型训练器，支持可选的 GradCache。
    """
```

### 构造函数

继承 `Trainer` 的所有参数，额外从 `args` 中提取 VLM2Vec 特定参数。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `use_grad_cache` | `bool` | 是否启用 GradCache |
| `gc_chunk_size` | `int` | GradCache 块大小 |
| `gc_sort_by_length` | `bool` | 是否在 GradCache 切块前按 query 有效长度升序对齐排序 |
| `max_length` | `int` | 最大序列长度 |

### 方法

#### training_step()

执行一个训练步骤。

```python
def training_step(
    self,
    model: nn.Module,
    inputs: dict[str, Any],
    num_items_in_batch: int | None = None,
) -> Tensor:
    """Perform a training step.

    执行一个训练步骤。

    Args:
        model: 模型
        inputs: 输入字典
        num_items_in_batch: batch 中的样本数

    Returns:
        损失张量
    """
```

**行为**：
- `use_grad_cache=False`：调用父类 `training_step`
- `use_grad_cache=True`：调用 `grad_cache_accumulate`，使用函数式 GradCache 流程

**GradCache 流程**：

1. 定义 encode 函数：`samples -> embeddings`
2. 定义损失函数：`DistributedContrastiveLoss`
3. 定义 forward/backward 函数：带梯度同步控制
4. 调用 `grad_cache_accumulate`
5. 当 `gc_sort_by_length=True` 时，在 GradCache 内按 query 有效长度升序对齐排序 query/positive groups 后再切 chunk

### 示例

```python
from vlm2emb.training.trainers import VLM2VecTrainer, VLM2VecTrainingArgs

# 创建 VLM2Vec 训练参数
args = VLM2VecTrainingArgs(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,

    # VLM2Vec 特定参数
    temperature=0.02,
    max_length=512,

    # GradCache 参数
    use_grad_cache=True,
    gc_chunk_size=4,
    gc_sort_by_length=True,
)

# 创建 VLM2VecTrainer
trainer = VLM2VecTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

# 训练
trainer.train()
```

---

## VLM2VecTrainingArgs

VLM2Vec 训练参数，继承自 HuggingFace TrainingArguments。

### 类定义

```python
from vlm2emb.training.trainers import VLM2VecTrainingArgs

@dataclass
class VLM2VecTrainingArgs(TrainingArguments):
    """Training arguments for VLM2Vec with embedding-specific parameters.

    VLM2Vec 训练参数，包含嵌入模型特定参数。
    """
```

### 参数

继承 `TrainingArguments` 的所有参数，额外添加：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | `float` | `0.02` | 对比损失温度 |
| `use_grad_cache` | `bool` | `False` | 是否使用 GradCache |
| `gc_chunk_size` | `int` | `4` | GradCache 块大小 |
| `gc_sort_by_length` | `bool` | `True` | 启用 GradCache 时，是否按 query 侧 `input_ids` 有效长度升序对齐排序 query/positive groups；设为 `False` 可保留原始 chunk 顺序 |
| `max_length` | `int` | `512` | 最大序列长度 |
| `interleave_batch_size` | `int` | `0` | BatchInterleaveSampler 的批次大小（0 禁用） |
| `interleave_stopping_strategy` | `str` | `"all_exhausted"` | 交错采样停止策略 |
| `interleave_shuffle_within_dataset` | `bool` | `True` | 是否打乱每个子数据集内部样本；为 `False` 时保持随机选择子数据集，但子数据集内部顺序读取 |

### 常用继承参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `output_dir` | `str` | 输出目录 |
| `per_device_train_batch_size` | `int` | 每设备训练 batch 大小 |
| `per_device_eval_batch_size` | `int` | 每设备评估 batch 大小 |
| `learning_rate` | `float` | 学习率 |
| `num_train_epochs` | `float` | 训练轮数 |
| `max_steps` | `int` | 最大训练步数 |
| `warmup_ratio` | `float` | 预热比例 |
| `warmup_steps` | `int` | 预热步数 |
| `weight_decay` | `float` | 权重衰减 |
| `bf16` | `bool` | 是否使用 BF16 |
| `fp16` | `bool` | 是否使用 FP16 |
| `gradient_accumulation_steps` | `int` | 梯度累积步数 |
| `save_strategy` | `str` | 保存策略（"no", "steps", "epoch"） |
| `save_steps` | `int` | 保存间隔步数 |
| `save_total_limit` | `int` | 最大检查点数量 |
| `logging_steps` | `int` | 日志间隔步数 |
| `eval_strategy` | `str` | 评估策略 |
| `eval_steps` | `int` | 评估间隔步数 |
| `dataloader_num_workers` | `int` | DataLoader 工作进程数 |
| `seed` | `int` | 随机种子 |

### 示例

```python
from vlm2emb.training.trainers import VLM2VecTrainingArgs

# 基础配置
args = VLM2VecTrainingArgs(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    temperature=0.02,
)

# 启用 GradCache 的配置
args = VLM2VecTrainingArgs(
    output_dir="./output",
    per_device_train_batch_size=32,  # 更大的 batch
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    temperature=0.02,
    use_grad_cache=True,
    gc_chunk_size=4,
    gc_sort_by_length=True,
)

# 完整配置
args = VLM2VecTrainingArgs(
    # 输出
    output_dir="./output/vlm2vec",

    # Batch 大小
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,

    # 学习率
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,

    # 训练轮数
    num_train_epochs=1,

    # 精度
    bf16=True,

    # 保存
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # 日志
    logging_steps=10,
    report_to=["tensorboard"],

    # VLM2Vec 特定
    temperature=0.02,
    max_length=512,

    # GradCache
    use_grad_cache=True,
    gc_chunk_size=4,
    gc_sort_by_length=True,

    # 交错数据集
    interleave_batch_size=8,
    interleave_stopping_strategy="all_exhausted",
    interleave_shuffle_within_dataset=False,

    # 其他
    dataloader_num_workers=4,
    seed=42,
)
```

---

## create_trainer

从配置创建训练器的工厂函数。

### 函数签名

```python
def create_trainer(
    config: Config | dict,
    model: PreTrainedModel,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    data_collator: Callable | None = None,
    processing_class: Any | None = None,
    **kwargs,
) -> Trainer:
    """Create trainer from configuration.

    从配置创建训练器。

    Args:
        config: 配置（OmegaConf Config 或 dict）
        model: 要训练的模型
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        data_collator: 数据整理器
        processing_class: Processor/Tokenizer
        **kwargs: 额外参数

    Returns:
        Trainer 实例
    """
```

### 配置格式

```yaml
training:
  trainer:
    type: vlm2vec_trainer  # AutoTrainer 查找

  args:
    output_dir: ./output
    per_device_train_batch_size: 8
    learning_rate: 1e-5
    # ... 其他参数
```

### 示例

```python
from vlm2emb import create_model
from vlm2emb.config import load_config
from vlm2emb.data import create_dataset, create_collator
from vlm2emb.training import create_trainer

# 加载配置
config = load_config("configs/experiments/vlm2vec.yaml")

# 创建组件
model = create_model(config)
dataset = create_dataset(config)
collator = create_collator(config)

# 创建 trainer
trainer = create_trainer(
    config=config,
    model=model,
    train_dataset=dataset,
    data_collator=collator,
)

# 训练
trainer.train()
```

---

## 注册表集成

### AutoTrainer

```python
from vlm2emb.auto import AutoTrainer

# 列出可用 Trainer
print(AutoTrainer.list_modules())
# ['trainer', 'vlm2vec_trainer']

# 从配置创建
trainer = AutoTrainer.from_config({
    "type": "vlm2vec_trainer",
    # ... 其他参数通过 args 传递
}, model=model, args=args, train_dataset=dataset)

# 注册自定义 Trainer
@AutoTrainer.register("my_trainer")
class MyTrainer(Trainer):
    pass
```

### AutoTrainingArgs

```python
from vlm2emb.auto import AutoTrainingArgs

# 列出可用 TrainingArgs
print(AutoTrainingArgs.list_modules())
# ['default', 'vlm2vec']

# 从配置创建
args = AutoTrainingArgs.from_config({
    "type": "vlm2vec",
    "output_dir": "./output",
    "temperature": 0.02,
})

# 注册自定义 TrainingArgs
@AutoTrainingArgs.register("my_args")
@dataclass
class MyTrainingArgs(TrainingArguments):
    custom_param: float = 0.1
```

---

## 完整示例

### 基础训练

```python
from vlm2emb import create_model
from vlm2emb.training import Trainer
from transformers import TrainingArguments

# 创建模型
model = create_model(config)

# 创建训练参数
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

# 训练
trainer.train()

# 保存
model.save_pretrained("./final_model")
```

### VLM2Vec + GradCache 训练

```python
from vlm2emb import create_model
from vlm2emb.training.trainers import VLM2VecTrainer, VLM2VecTrainingArgs

# 创建模型
model = create_model(config)

# 创建 VLM2Vec 训练参数
args = VLM2VecTrainingArgs(
    output_dir="./output",
    per_device_train_batch_size=32,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,

    # VLM2Vec 参数
    temperature=0.02,
    max_length=512,

    # GradCache
    use_grad_cache=True,
    gc_chunk_size=4,
    gc_sort_by_length=True,
)

# 创建 VLM2VecTrainer
trainer = VLM2VecTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

# 训练
trainer.train()

# 保存
model.save_pretrained("./final_model")
```

### 分布式训练

```bash
# 使用 Accelerate
accelerate launch scripts/train.py configs/experiments/vlm2vec.yaml

# 使用 DeepSpeed
accelerate launch scripts/train.py configs/experiments/vlm2vec.yaml \
    train.args.deepspeed=path/to/deepspeed_config.json
```

---

## 相关文档

- [训练系统](../architecture/training-system.md) - 训练系统架构设计
- [BToks API](./model.md) - 模型 API 参考
