"""VLM2Vec trainer implementation with GradCache support.

This module provides the VLM2VecTrainer for training VLM2Vec embedding models.
It extends Trainer with optional GradCache support for large batch training.

Reference:
    VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks
    https://arxiv.org/abs/2410.05160
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, TrainingArguments

from vlm2emb.auto import AutoTrainer, AutoTrainingArgs
from vlm2emb.data.processors import batch_processor_outputs
from vlm2emb.training.trainer import Trainer

logger = logging.getLogger(__name__)


# ============================================================================
# VLM2Vec training arguments.
# ============================================================================


@AutoTrainingArgs.register("vlm2vec")
@dataclass
class VLM2VecTrainingArgs(TrainingArguments):
    """Training arguments for VLM2Vec with embedding-specific parameters.

    Extends HuggingFace TrainingArguments with:
    - temperature: Contrastive loss temperature
    - use_grad_cache: Whether to use GradCache
    - gc_chunk_size: Chunk size for GradCache
    - gc_sort_by_length: Whether GradCache sorts aligned groups by query length
    - max_length: Maximum sequence length
    """

    temperature: float = field(
        default=0.02,
        metadata={"help": "Temperature for contrastive loss."},
    )
    use_grad_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use GradCache for large-batch training."},
    )
    gc_chunk_size: int = field(
        default=4,
        metadata={"help": "Chunk size for GradCache query and passage groups."},
    )
    gc_sort_by_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether GradCache sorts aligned input groups by query input_ids "
                "effective length before chunking."
            )
        },
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    interleave_batch_size: int = field(
        default=0,
        metadata={
            "help": "Batch size for BatchInterleaveSampler. If > 0, uses BatchInterleaveSampler "
            "for batch-wise interleaving of multi-source datasets. "
            "Set to 0 to use default RandomSampler."
        },
    )
    interleave_stopping_strategy: str = field(
        default="all_exhausted",
        metadata={
            "help": "Stopping strategy for BatchInterleaveSampler. "
            "'all_exhausted': cycle short datasets until longest is exhausted. "
            "'first_exhausted': stop when any dataset is exhausted."
        },
    )
    interleave_shuffle_within_dataset: bool = field(
        default=True,
        metadata={
            "help": "Whether to shuffle samples within each sub-dataset when using "
            "BatchInterleaveSampler. If False, sub-dataset selection remains random "
            "but samples are read sequentially within each sub-dataset."
        },
    )


# ============================================================================
# VLM2Vec trainer.
# ============================================================================


@AutoTrainer.register("vlm2vec_trainer")
class VLM2VecTrainer(Trainer):
    """Trainer for VLM2Vec embedding model with optional GradCache.

    This trainer extends Trainer with:
    - Optional GradCache for memory-efficient large batch training
    - VLM2Vec specific training step implementations

    GradCache design:
        GradCache uses a functional API (grad_cache_accumulate) that:
        1. Splits inputs into chunks for memory efficiency
        2. Computes representations without gradients
        3. Builds gradient cache from full-batch loss
        4. Backward through each chunk immediately to release memory

    Attributes:
        use_grad_cache: Whether GradCache is enabled.
        gc_chunk_size: Chunk size for GradCache.
        max_length: Maximum sequence length.

    Example:
        >>> from vlm2emb.training.trainers import VLM2VecTrainer, VLM2VecTrainingArgs
        >>>
        >>> args = VLM2VecTrainingArgs(
        ...     output_dir="./output",
        ...     per_device_train_batch_size=8,
        ...     temperature=0.02,
        ...     use_grad_cache=True,
        ... )
        >>> trainer = VLM2VecTrainer(
        ...     model=model,
        ...     args=args,
        ...     train_dataset=dataset,
        ...     data_collator=collator,
        ... )
        >>> trainer.train()
    """

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
        """Initialize VLM2VecTrainer.

        Args:
            model: The model to train
            args: Training arguments (VLM2VecTrainingArgs with temperature, use_grad_cache, etc.)
            data_collator: Data collator for batching
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Processor/tokenizer for the model
            model_init: Function to initialize model
            compute_metrics: Function to compute metrics
            callbacks: Training callbacks
            optimizers: Tuple of (optimizer, scheduler)
            preprocess_logits_for_metrics: Function to preprocess logits
            **kwargs: Additional arguments (for future extensibility)
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )

        # Get VLM2Vec-specific args from TrainingArguments.
        self.max_length = getattr(self.args, "max_length", 512)
        self.use_grad_cache = getattr(self.args, "use_grad_cache", False)
        self.gc_chunk_size = getattr(self.args, "gc_chunk_size", 4)
        self.gc_sort_by_length = bool(getattr(self.args, "gc_sort_by_length", True))

        if self.use_grad_cache:
            logger.info(
                "GradCache enabled with "
                f"chunk_size={self.gc_chunk_size}, "
                f"sort_by_length={self.gc_sort_by_length}",
            )

    # ========================================================================
    # Training step override.
    # ========================================================================

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> Tensor:
        """Perform a training step.

        Only overrides parent when GradCache is enabled.
        When GradCache is disabled, delegates to HuggingFace Trainer's training_step.

        Args:
            model: The model
            inputs: Dict with "query", "positive", "negative" (optional), "metadata" keys
            num_items_in_batch: Number of items in batch

        Returns:
            Loss tensor
        """
        from vlm2emb.training.grad_cache import grad_cache_accumulate
        from vlm2emb.training.losses.contrastive import DistributedContrastiveLoss

        # GradCache mode: custom training step.
        model.train()

        # Extract query and positive groups from the collated batch.
        qry_inputs = inputs["query"]
        pos_inputs = inputs["positive"]

        # Define loss function; it uses distributed gather internally.
        loss_fn = DistributedContrastiveLoss(
            temperature=getattr(self.args, "temperature", 0.02),
            scale_loss=True,
        )

        # Create closures for GradCache phases
        encode_fn = self._make_encode_fn(model)
        forward_backward_fn = self._make_forward_backward_fn(model)

        # Run grad_cache_accumulate
        loss, _ = grad_cache_accumulate(
            inputs=[qry_inputs, pos_inputs],
            encode_fns=[encode_fn, encode_fn],
            loss_fn=loss_fn,
            chunk_sizes=[self.gc_chunk_size, self.gc_chunk_size],
            forward_backward_fns=[forward_backward_fn, forward_backward_fn],
            aligned_sort_key_fn=(
                self._input_token_length if self.gc_sort_by_length else None
            ),
            aligned_sort_group_idx=0,
        )

        # Normalize: DistributedContrastiveLoss returns CE_mean_global × W
        # (scale_loss=True compensates DDP ÷W for gradient correctness).
        # Divide by W so the returned loss = CE_mean_global, invariant to W.
        return loss / self.accelerator.num_processes

    @staticmethod
    def _input_token_length(sample: dict[str, Any]) -> int:
        """Return effective token length for one processed sample."""
        attention_mask = sample.get("attention_mask")
        if attention_mask is not None:
            if torch.is_tensor(attention_mask):
                return int(attention_mask.long().sum().item())
            if hasattr(attention_mask, "sum"):
                return int(attention_mask.sum())
            return int(sum(attention_mask))

        input_ids = sample.get("input_ids")
        if input_ids is None:
            raise ValueError("Cannot sort sample by length without input_ids")
        if torch.is_tensor(input_ids):
            return int(input_ids.numel())
        if (
            isinstance(input_ids, (list, tuple))
            and len(input_ids) == 1
            and isinstance(input_ids[0], (list, tuple))
        ):
            return len(input_ids[0])
        if isinstance(input_ids, (list, tuple)):
            return len(input_ids)
        return 1

    def _make_encode_fn(self, model: nn.Module) -> Callable[[list[dict]], Tensor]:
        """Build GradCache phase-A encode closure.

        Creates a closure that encodes samples into embeddings using autocast.

        Args:
            model: The model to use for encoding

        Returns:
            Encode function: samples -> embeddings Tensor
        """
        wrapper = self.processing_class

        def encode_fn(samples: list[dict]) -> Tensor:
            """Encode a list of samples into embeddings."""
            with self.accelerator.autocast():
                batch = batch_processor_outputs(samples, wrapper)
                batch = self._prepare_inputs(batch)
                with self.accelerator.no_sync(model):
                    outputs = model(**batch)
                return outputs["embeddings"]

        return encode_fn

    def _make_forward_backward_fn(
        self, model: nn.Module
    ) -> Callable[[list[dict], Tensor, Any], None]:
        """Build GradCache phase-B forward_backward closure.

        Creates a closure that performs forward pass and backward pass,
        with autocast wrapping only forward + surrogate, and backward
        running outside autocast per PyTorch best practice.

        Args:
            model: The model to use for forward pass

        Returns:
            Forward-backward function: (chunk, cache, ctx) -> None
        """
        from vlm2emb.training.grad_cache import ChunkContext

        wrapper = self.processing_class

        def forward_backward_fn(
            chunk: list[dict], cache: Tensor, ctx: ChunkContext
        ) -> None:
            """Forward + backward with DDP sync control."""
            is_last = (
                ctx.group_idx == ctx.num_groups - 1
                and ctx.chunk_idx == ctx.num_chunks - 1
            )

            sync_ctx = nullcontext() if is_last else self.accelerator.no_sync(model)

            with sync_ctx:
                # autocast wraps forward + surrogate only; backward runs outside
                # per PyTorch best practice (autograd handles dtype automatically)
                with self.accelerator.autocast():
                    batch = batch_processor_outputs(chunk, wrapper)
                    batch = self._prepare_inputs(batch)

                    reps = model(**batch)["embeddings"]

                    surrogate = torch.dot(reps.flatten(), cache.flatten())
                self.accelerator.backward(surrogate)

        return forward_backward_fn
