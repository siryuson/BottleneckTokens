"""Pooling modules for the VLM2Emb pipeline.

This module provides pooling strategies to convert variable-length
hidden state sequences into fixed-size embedding vectors.

Available pooling strategies:
    - LastTokenPooling: Extract last valid token (VLM2Vec default)
    - MeanPooling: Mean over all valid tokens

Output keys:
    - embeddings: (B, D) - Pooled embedding vector
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LastTokenPooling(nn.Module):
    """Last token pooling module (VLM2Vec default).

    Extracts the hidden state at the last valid token position.
    For left-padded sequences, this is simply the last position.
    For right-padded sequences, finds the last non-padded position.

    Example:
        >>> pooling = LastTokenPooling()
        >>> output = pooling(last_hidden_state=hidden, attention_mask=mask)
        >>> embeddings = output["embeddings"]  # (B, D)
    """

    def __init__(self, **kwargs):
        """Initialize LastTokenPooling."""
        super().__init__()

    @classmethod
    def from_config(cls, config: dict) -> "LastTokenPooling":
        """Create from configuration dict."""
        config = config.copy()
        config.pop("type", None)
        return cls(**config)

    def forward(
        self,
        last_hidden_state: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Extract embedding from last valid token.

        Args:
            last_hidden_state: Hidden states (B, L, D)
            attention_mask: Attention mask (B, L), 1 for valid, 0 for padding
            **kwargs: Passthrough arguments

        Returns:
            Dict with keys:
                - embeddings: (B, D)
                - plus all kwargs passed through
        """
        batch_size, seq_len, hidden_size = last_hidden_state.shape

        if attention_mask is not None:
            left_padding = attention_mask[:, -1].sum() == batch_size

            if left_padding:
                pooled = last_hidden_state[:, -1, :]
            else:
                eos_indices = attention_mask.sum(dim=1).long() - 1
                pooled = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    eos_indices,
                ]
        else:
            pooled = last_hidden_state[:, -1, :]

        return {
            "embeddings": pooled,
            "last_hidden_state": last_hidden_state,
            "attention_mask": attention_mask,
            **kwargs,
        }


class MeanPooling(nn.Module):
    """Mean pooling module.

    Computes mean of all valid token hidden states, respecting attention mask.

    Example:
        >>> pooling = MeanPooling()
        >>> output = pooling(last_hidden_state=hidden, attention_mask=mask)
        >>> embeddings = output["embeddings"]  # (B, D)
    """

    def __init__(self, **kwargs):
        """Initialize MeanPooling."""
        super().__init__()

    @classmethod
    def from_config(cls, config: dict) -> "MeanPooling":
        """Create from configuration dict."""
        config = config.copy()
        config.pop("type", None)
        return cls(**config)

    def forward(
        self,
        last_hidden_state: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute mean pooling over sequence.

        Args:
            last_hidden_state: Hidden states (B, L, D)
            attention_mask: Attention mask (B, L)
            **kwargs: Passthrough arguments

        Returns:
            Dict with keys:
                - embeddings: (B, D)
                - plus all kwargs passed through
        """
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            pooled = sum_hidden / sum_mask.clamp(min=1e-9)
        else:
            pooled = last_hidden_state.mean(dim=1)

        return {
            "embeddings": pooled,
            "last_hidden_state": last_hidden_state,
            "attention_mask": attention_mask,
            **kwargs,
        }


class Normalize(nn.Module):
    """L2 normalization module.

    Applies L2 normalization to sentence embeddings.

    Example:
        >>> normalize = Normalize()
        >>> output = normalize(embeddings=emb)
        >>> normalized_emb = output["embeddings"]  # L2-normalized
    """

    def __init__(self, **kwargs):
        """Initialize Normalize."""
        super().__init__()

    @classmethod
    def from_config(cls, config: dict) -> "Normalize":
        """Create from configuration dict."""
        config = config.copy()
        config.pop("type", None)
        return cls(**config)

    def forward(
        self,
        embeddings: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Apply L2 normalization.

        In generation mode (loss present in features), skip normalization
        and pass through all features unchanged.

        Args:
            embeddings: Embedding vectors (B, D)
            **kwargs: Passthrough arguments

        Returns:
            Dict with keys:
                - embeddings: (B, D) - L2 normalized
                - plus all kwargs passed through
        """
        if embeddings is None:
            raise ValueError("Normalize requires embeddings input")

        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return {
            "embeddings": normalized,
            **kwargs,
        }
