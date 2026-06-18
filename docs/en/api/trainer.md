# Trainer API Reference

> **Version**: 0.1
> **Updated**: 2026-01-25
> **Module**: `vlm2emb.training.trainer`

This document provides detailed reference for BToks Trainer API.

---

## Table of Contents

1. [Trainer](#trainer) - Base trainer class
2. [VLM2VecTrainer](#vlm2vectrainer) - VLM2Vec-specific trainer

---

## Trainer

Base trainer class extending HuggingFace Trainer for contrastive learning.

### Class Definition

```python
from vlm2emb.training import Trainer

class Trainer(transformers.Trainer):
    """Trainer for contrastive learning with multimodal embeddings.

    Extends HuggingFace Trainer to support (query, positive) pair training.
    """
```

### Constructor

```python
def __init__(
    self,
    model: PreTrainedModel | nn.Module,
    args: TrainingArguments | VLM2VecTrainingArgs,
    train_dataset: Dataset | IterableDataset,
    data_collator: Callable | None = None,
    eval_dataset: Dataset | dict[str, Dataset] | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_init: Callable[[], PreTrainedModel] | None = None,
    compute_metrics: Callable[[EvalPrediction], dict] | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[torch.optim.Optimizer, LRScheduler] | None = None,
    preprocess_logits_for_metrics: Callable[[Tensor, Tensor] | None = None,
):
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `PreTrainedModel \| nn.Module` | Model to train |
| `args` | `TrainingArguments` | Training arguments |
| `train_dataset` | `Dataset` | Training dataset |
| `data_collator` | `Callable` | Data collator function |
| `eval_dataset` | `Dataset` | Evaluation dataset |
| `tokenizer` | `PreTrainedTokenizerBase` | Tokenizer for preprocessing |
| `model_init` | `Callable` | Model initialization function |
| `compute_metrics` | `Callable` | Metrics computation function |
| `callbacks` | `list[TrainerCallback]` | Custom callbacks |
| `optimizers` | `tuple` | Custom optimizers |
| `preprocess_logits_for_metrics` | `Callable` | Logits preprocessing for metrics |

### Core Methods

#### compute_loss

Compute contrastive loss for (query, positive) pairs.

```python
def compute_loss(
    self,
    model: nn.Module,
    inputs: dict[str, dict],
    return_outputs: bool = False,
    num_items_in_batch: int | None = None,
) -> Tensor | tuple[Tensor, Any]:
```

**Parameters**:
- `model`: Model being trained
- `inputs`: Dict containing `"query"` and `"positive"` keys
- `return_outputs`: Whether to return model outputs
- `num_items_in_batch`: Number of samples in batch

**Returns**: Loss tensor, optionally with outputs

**Input Format**:
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
    "metadata": list[dict],          # Optional
}
```

**Example**:
```python
# Custom compute_loss override
def compute_loss(self, model, inputs, return_outputs=False):
    qry_inputs = inputs["query"]
    pos_inputs = inputs["positive"]

    qry_outputs = model(**qry_inputs)
    pos_outputs = model(**pos_inputs)

    qry_embeds = qry_outputs["embeddings"]
    pos_embeds = pos_outputs["embeddings"]

    loss = self.loss_fn(qry_embeds, pos_embeds)

    if return_outputs:
        return loss, {"qry_embeds": qry_embeds, "pos_embeds": pos_embeds}
    return loss
```

#### _setup_loss_fn

Setup contrastive loss function based on distributed environment.

```python
def _setup_loss_fn(self) -> None:
    """Set up contrastive loss function.

    Uses DistributedContrastiveLoss in distributed mode,
    ContrastiveLoss in single GPU mode.
    """
```

**Behavior**:
- Single GPU: Uses `ContrastiveLoss`
- Multi-GPU: Uses `DistributedContrastiveLoss` with `all_gather`

---

## VLM2VecTrainer

Trainer with GradCache support for large batch training.

### Class Definition

```python
from vlm2emb.training.trainers import VLM2VecTrainer

class VLM2VecTrainer(Trainer):
    """Trainer with GradCache support for VLM2Vec training.

    Supports memory-efficient large batch training through gradient caching.
    """
```

### Constructor

```python
def __init__(
    self,
    model: PreTrainedModel | nn.Module,
    args: VLM2VecTrainingArgs,
    train_dataset: Dataset | IterableDataset,
    data_collator: Callable | None = None,
    eval_dataset: Dataset | dict[str, Dataset] | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_init: Callable[[], PreTrainedModel] | None = None,
    compute_metrics: Callable[[EvalPrediction], dict] | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[torch.optim.Optimizer, LRScheduler] | None = None,
    preprocess_logits_for_metrics: Callable[[Tensor, Tensor] | None = None,
):
```

### VLM2VecTrainingArgs

Specialized training arguments for VLM2Vec.

```python
from vlm2emb.training.trainers import VLM2VecTrainingArgs

@dataclass
class VLM2VecTrainingArgs(TrainingArguments):
    """Training arguments for VLM2Vec training."""
```

#### Additional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `temperature` | `float` | `0.02` | Temperature for contrastive loss |
| `max_length` | `int` | `512` | Maximum sequence length |
| `use_grad_cache` | `bool` | `False` | Enable GradCache |
| `gc_chunk_size` | `int` | `4` | GradCache chunk size |
| `gc_sort_by_length` | `bool` | `True` | When GradCache is enabled, sort aligned query/positive groups by ascending effective query `input_ids` length before chunking. Set to `False` to preserve the original chunk order |
| `interleave_batch_size` | `int` | `0` | Batch size for BatchInterleaveSampler |
| `interleave_stopping_strategy` | `str` | `"all_exhausted"` | Stopping strategy for interleaved dataset sampling |
| `interleave_shuffle_within_dataset` | `bool` | `True` | Whether to shuffle samples within each sub-dataset while keeping sub-dataset selection random |

### training_step

Training step with optional GradCache support.

```python
def training_step(
    self,
    model: nn.Module,
    inputs: dict[str, dict],
) -> Tensor:
```

**With GradCache**:
```python
def training_step(self, model, inputs):
    if self.args.use_grad_cache:
        # Use GradCache for memory-efficient training
        loss = grad_cache_accumulate(
            inputs=[inputs["query"], inputs["positive"]],
            encode_fns=[encode_fn, encode_fn],
            loss_fn=self.loss_fn,
            chunk_sizes=[self.args.gc_chunk_size, self.args.gc_chunk_size],
            aligned_sort_key_fn=query_length_key if self.args.gc_sort_by_length else None,
        )
        return loss.detach()
    else:
        # Standard training step
        return super().training_step(model, inputs)
```

### Example Usage

```python
from vlm2emb.training.trainers import VLM2VecTrainer, VLM2VecTrainingArgs

# Create training arguments
args = VLM2VecTrainingArgs(
    output_dir="./output",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,

    # VLM2Vec-specific parameters
    temperature=0.02,
    max_length=512,

    # GradCache configuration
    use_grad_cache=True,
    gc_chunk_size=4,
    gc_sort_by_length=True,

    # Interleaved dataset sampling
    interleave_batch_size=8,
    interleave_stopping_strategy="all_exhausted",
    interleave_shuffle_within_dataset=False,
)

# Create trainer
trainer = VLM2VecTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# Train
trainer.train()

# Save model
model.save_pretrained(args.output_dir)
```

---

## Loss Functions

### ContrastiveLoss

Basic InfoNCE contrastive loss.

```python
from vlm2emb.training.losses import ContrastiveLoss

loss_fn = ContrastiveLoss(temperature=0.02)
loss = loss_fn(query_embeddings, positive_embeddings)
```

### DistributedContrastiveLoss

Distributed contrastive loss with cross-GPU embedding gathering.

```python
from vlm2emb.training.losses import DistributedContrastiveLoss

loss_fn = DistributedContrastiveLoss(
    temperature=0.02,
    scale_loss=True,  # Scale loss by world_size
)
```

---

## Related Documents

- [Training System](../architecture/training-system.md) - Training system overview
- [GradCache](../architecture/training-system.md#6-gradcache) - GradCache details
- [Model API](./model.md) - Model API Reference
