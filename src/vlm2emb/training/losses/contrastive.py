"""Contrastive loss functions for embedding learning.

Implements InfoNCE and variants for multimodal embedding training.
"""


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from torch import Tensor

from vlm2emb.auto import AutoLoss


@AutoLoss.register("ContrastiveLoss")
class ContrastiveLoss(nn.Module):
    """Simple contrastive loss (InfoNCE).

    Computes bidirectional contrastive loss between queries and keys.
    For each query, the corresponding key should be the positive example.

    Args:
        temperature: Temperature parameter for softmax (default: 0.02)
        learnable_temperature: Whether temperature is learnable (default: False)

    Example:
        >>> loss_fn = ContrastiveLoss(temperature=0.02)
        >>> query_embeds = model.encode(queries)  # (B, D)
        >>> key_embeds = model.encode(keys)  # (B, D)
        >>> loss = loss_fn(query_embeds, key_embeds)
    """

    def __init__(
        self,
        temperature: float = 0.02,
        learnable_temperature: bool = False,
    ):
        """Initialize contrastive loss."""
        super().__init__()

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        target: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute contrastive loss.

        Args:
            query: Query embeddings (B, D)
            key: Key embeddings (B * K, D) where K is number of keys per query
            target: Optional target indices (B,). If None, assumes 1:1 matching
                   with target[i] = i * (K/B) for uniform K keys per query
            reduction: Loss reduction ('mean', 'sum', or 'none')

        Returns:
            Scalar loss value or per-sample losses if reduction='none'

        Note:
            - Query and key should be L2-normalized for best results
            - Default assumes K keys per query are contiguous in key tensor
        """
        # Compute similarity matrix (B, B*K)
        logits = torch.matmul(query, key.transpose(0, 1))

        # Scale by temperature
        logits = logits / self.temperature

        # Create target labels if not provided
        if target is None:
            # Assume keys are organized as [key_0_0, ..., key_0_K, key_1_0, ...]
            # where key_i_j is the j-th key for query i
            target_per_query = key.size(0) // query.size(0)
            target = torch.arange(
                0,
                query.size(0) * target_per_query,
                target_per_query,
                device=query.device,
                dtype=torch.long,
            )

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, target, reduction=reduction)

        return loss


@AutoLoss.register("DistributedContrastiveLoss")
class DistributedContrastiveLoss(ContrastiveLoss):
    """Distributed contrastive loss with cross-GPU gathering.

    Gathers embeddings from all GPUs to compute contrastive loss across
    the full global batch. This is crucial for large-scale embedding training.

    Gradient support:
        Uses the `gathered[rank] = tensor` trick to preserve gradient flow.
        dist.all_gather itself does not support gradients, but by replacing
        the current rank's position with the original tensor, autograd can
        correctly compute gradients for the local tensor.

    Args:
        temperature: Temperature parameter (default: 0.02)
        learnable_temperature: Whether temperature is learnable
        scale_loss: Whether to scale loss by world size (default: True)

    Example:
        >>> # In distributed training setup
        >>> loss_fn = DistributedContrastiveLoss(temperature=0.02)
        >>> query_embeds = model.encode(queries)  # Local batch (B_local, D)
        >>> key_embeds = model.encode(keys)  # Local batch (B_local, D)
        >>> loss = loss_fn(query_embeds, key_embeds)  # Uses global batch

        >>> # With grad_cache_accumulate
        >>> loss = grad_cache_accumulate(
        ...     inputs=[queries, passages],
        ...     forward_fns=[forward_fn, forward_fn],
        ...     loss_fn=loss_fn,  # Handles gather internally
        ...     chunk_sizes=[4, 4],
        ... )
    """

    def __init__(
        self,
        temperature: float = 0.02,
        learnable_temperature: bool = False,
        scale_loss: bool = True,
    ):
        """Initialize distributed contrastive loss."""
        super().__init__(temperature, learnable_temperature)

        self.scale_loss = scale_loss

        # Get distributed process metadata from accelerate PartialState.
        state = PartialState()
        self.world_size = state.num_processes
        self.rank = state.process_index

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        target: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute distributed contrastive loss.

        Args:
            query: Local query embeddings (B_local, D)
            key: Local key embeddings (B_local, D)
            target: Optional local targets (B_local,)
            reduction: Loss reduction

        Returns:
            Scalar loss value
        """
        # Gather embeddings from all processes.
        if self.world_size > 1:
            gathered_query = self._gather_tensor(query)
            gathered_key = self._gather_tensor(key)

            # Adjust targets for global indexing.
            if target is not None:
                offset = self.rank * query.size(0)
                gathered_target = target + offset
            else:
                gathered_target = None
        else:
            gathered_query = query
            gathered_key = key
            gathered_target = target

        # Compute loss on gathered tensors
        loss = super().forward(gathered_query, gathered_key, gathered_target, reduction)

        # Scale loss by world size if requested.
        if self.scale_loss and self.world_size > 1:
            loss = loss * self.world_size

        return loss

    def _gather_tensor(self, tensor: Tensor) -> Tensor:
        """Gather tensor from all processes with gradient support.

        The key insight is that dist.all_gather does NOT support gradients,
        but by replacing gathered[rank] with the original tensor, we preserve
        the computation graph for the local tensor. This allows autograd to
        correctly compute gradients.

        Args:
            tensor: Local tensor (B_local, D)

        Returns:
            Gathered tensor (B_global, D) where B_global = B_local * world_size
        """
        if self.world_size == 1:
            return tensor

        # Use empty_like instead of zeros_like because all_gather overwrites every entry.
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)

        # Key detail: replace this rank's gathered tensor with the original tensor
        # so the local computation graph remains connected for autograd.
        gathered[self.rank] = tensor

        return torch.cat(gathered, dim=0)


@AutoLoss.register("InBatchContrastiveLoss")
class InBatchContrastiveLoss(nn.Module):
    """In-batch contrastive loss for classification-style tasks.

    Used when each query has multiple candidate keys (e.g., multiple choice)
    and we want to classify which one is correct.

    Args:
        temperature: Temperature parameter (default: 1.0)
        n_hard_negatives: Number of hard negatives per query (default: 0)

    Example:
        >>> loss_fn = InBatchContrastiveLoss(temperature=1.0)
        >>> query_embeds = model.encode(queries)  # (B, D)
        >>> # Multiple keys per query
        >>> key_embeds = model.encode(keys)  # (B, K, D)
        >>> loss = loss_fn(query_embeds, key_embeds)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        n_hard_negatives: int = 0,
    ):
        """Initialize in-batch contrastive loss."""
        super().__init__()
        self.temperature = temperature
        self.target_per_query = n_hard_negatives + 1

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Compute in-batch contrastive loss.

        Args:
            query: Query embeddings (B, D)
            key: Key embeddings (B, K, D) where K is number of candidates
            reduction: Loss reduction

        Returns:
            Tuple of (loss, loss_details) where loss_details contains:
                - logits: Similarity scores (B, K)
                - labels: Target labels (B,)
                - preds: Predicted labels (B,)
        """
        batch_size, embed_dim = query.size()
        # Compute similarity: (B, 1, D) @ (B, D, K) -> (B, 1, K) -> (B, K)
        logits = torch.einsum("bd,bkd->bk", query, key)  # (B, K)

        # Scale by temperature
        logits = logits * self.temperature

        # Target is always first candidate (index 0)
        target = torch.zeros(batch_size, dtype=torch.long, device=query.device)

        # Compute loss
        loss = F.cross_entropy(logits, target, reduction=reduction)

        # Get predictions
        preds = torch.argmax(logits, dim=-1)

        # Return loss and details
        loss_details = {
            "logits": logits,
            "labels": target,
            "preds": preds,
            "accuracy": (preds == target).float().mean(),
        }

        return loss, loss_details


def compute_contrastive_loss(
    query_embeds: Tensor,
    key_embeds: Tensor,
    temperature: float = 0.02,
    distributed: bool = False,
) -> Tensor:
    """Convenience function to compute contrastive loss.

    Args:
        query_embeds: Query embeddings (B, D)
        key_embeds: Key embeddings (B, D)
        temperature: Temperature parameter
        distributed: Whether to use distributed version

    Returns:
        Scalar loss value

    Example:
        >>> loss = compute_contrastive_loss(
        ...     query_embeds,
        ...     key_embeds,
        ...     temperature=0.02,
        ...     distributed=True,  # Auto-detects if distributed is available
        ... )
    """
    if distributed:
        state = PartialState()
        if state.num_processes > 1:
            loss_fn = DistributedContrastiveLoss(temperature=temperature)
        else:
            loss_fn = ContrastiveLoss(temperature=temperature)
    else:
        loss_fn = ContrastiveLoss(temperature=temperature)

    return loss_fn(query_embeds, key_embeds)
