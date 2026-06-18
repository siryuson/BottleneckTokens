"""BToks modules for the VLM2Emb pipeline.

This module provides BToksTokenInjector and BToksPooling for the BToks
bottleneck token injection approach. Tokens are real vocabulary entries
registered via tokenizer, not nn.Parameter.

Pipeline: BToksTokenInjector -> Backbone -> BToksPooling -> Normalize
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Prefix for btoks special tokens in tokenizer
BToks_TOKEN_PREFIX = "<|btok_"
BToks_TOKEN_SUFFIX = "|>"


def _btoks_token_name(i: int) -> str:
    """Generate btoks token name like <|btok_0|>, <|btok_1|>, ..."""
    return f"{BToks_TOKEN_PREFIX}{i}{BToks_TOKEN_SUFFIX}"


def get_vision_token_ids(config) -> list[int]:
    """Auto-discover vision token IDs from HF model config.

    Checks common config attributes used by various VLMs:
    - image_token_id (Qwen2-VL, Qwen3-VL)
    - video_token_id (Qwen2-VL, Qwen3-VL)
    - image_token_index (LLaVA)

    Args:
        config: HuggingFace model config object.

    Returns:
        List of vision token IDs found in config. Empty list if none found.
    """
    VISION_TOKEN_ATTRS = [
        "image_token_id",
        "video_token_id",
        "image_token_index",
        "video_token_index",
    ]
    token_ids = []
    for attr in VISION_TOKEN_ATTRS:
        tid = getattr(config, attr, None)
        if tid is not None and isinstance(tid, int):
            token_ids.append(tid)
    return token_ids


def extract_btoks_kv(
    past_key_values: Any,
    btoks_token_mask: Tensor,
) -> Any | None:
    """Extract KV cache entries at btoks token positions.

    Returns a DynamicCache containing only the btoks-token KV entries so that
    downstream model forward passes (which call ``get_seq_length``, etc.) work
    correctly.

    Args:
        past_key_values: KV cache — DynamicCache or tuple/list of (key, value).
        btoks_token_mask: Boolean mask with shape (B, L), True at btoks positions.

    Returns:
        DynamicCache restricted to btoks positions, or None when
        btoks_token_mask has no True values.
    """
    positions = _positions_from_mask(btoks_token_mask)
    if positions is None:
        return None

    return extract_kv_at_positions(
        past_key_values=past_key_values,
        positions=positions,
    )


def _positions_from_mask(mask: Tensor) -> Tensor | None:
    """Return per-sample token positions from a boolean mask.

    Args:
        mask: Boolean mask with shape (B, L).

    Returns:
        Long tensor with shape (B, S), or None when no positions are selected.

    Raises:
        ValueError: If selected position count differs across samples.
    """
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, L)")

    mask = mask.bool()
    counts = mask.sum(dim=1)
    if counts.numel() == 0 or int(counts.max().item()) == 0:
        return None
    if not torch.equal(counts, counts[0].expand_as(counts)):
        raise ValueError("Selected KV positions must have the same count per sample")

    positions = mask.nonzero(as_tuple=False)[:, 1].view(mask.shape[0], int(counts[0].item()))
    return positions.to(device=mask.device, dtype=torch.long)


def last_valid_token_indices(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tensor:
    """Return the final valid token index for each sample.

    Uses the last non-zero entry in ``attention_mask``. When no mask is
    available, or a row has no valid entries, falls back to the sequence tail.
    """
    if attention_mask is None:
        return torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (B, L)")
    if attention_mask.shape[0] != batch_size or attention_mask.shape[1] != seq_len:
        raise ValueError(
            "attention_mask shape must match batch and sequence dimensions"
        )

    valid = attention_mask.to(device=device).ne(0)
    position_values = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)
    fallback = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
    return torch.where(
        valid.any(dim=1),
        valid.long().mul(position_values).max(dim=1).values,
        fallback,
    )


def pool_last_valid_token(
    last_hidden_state: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Pool hidden states at the final valid token position."""
    batch_size, seq_len, _ = last_hidden_state.shape
    indices = last_valid_token_indices(
        attention_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        device=last_hidden_state.device,
    )
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        indices,
    ]


def extract_last_token_kv(
    past_key_values: Any,
    attention_mask: Tensor | None,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Any:
    """Extract one KV cache entry per sample at the final valid token."""
    indices = last_valid_token_indices(
        attention_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    return extract_kv_at_positions(
        past_key_values=past_key_values,
        positions=indices.view(batch_size, 1),
    )


def extract_kv_at_positions(
    past_key_values: Any,
    positions: Tensor,
) -> Any:
    """Extract KV cache entries at per-sample token positions.

    Args:
        past_key_values: DynamicCache or tuple/list of (key, value) layers.
        positions: Long tensor with shape (B, S), where each row contains the
            sequence positions to keep for that sample.

    Returns:
        DynamicCache restricted to the requested positions.
    """
    from transformers import DynamicCache

    if positions.ndim != 2:
        raise ValueError("positions must have shape (B, S)")
    positions = positions.long()

    # Collect (key, value) per layer from any supported format
    layers: list[tuple[Tensor, Tensor]] = []

    def _select_seq(key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        if key_states.ndim < 4 or value_states.ndim < 4:
            raise ValueError("KV cache tensors must have shape (B, H, L, D)")
        if key_states.shape[0] != positions.shape[0]:
            raise ValueError("positions batch size must match KV cache batch size")

        pos = positions.to(device=key_states.device)
        key_index = pos[:, None, :, None].expand(
            -1,
            key_states.shape[1],
            -1,
            key_states.shape[3],
        )
        value_index = pos[:, None, :, None].expand(
            -1,
            value_states.shape[1],
            -1,
            value_states.shape[3],
        )
        return (
            key_states.gather(dim=2, index=key_index),
            value_states.gather(dim=2, index=value_index),
        )

    if hasattr(past_key_values, "layers") or (
        hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache")
    ):
        # DynamicCache-like: __iter__ yields (key_states, value_states)
        for key_states, value_states in past_key_values:
            layers.append(_select_seq(key_states, value_states))
    elif isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("Invalid tuple/list past_key_values layer format")
            layers.append(_select_seq(layer[0], layer[1]))
    else:
        raise TypeError("Unsupported past_key_values format")

    # Build a proper DynamicCache so downstream model code can call
    # get_seq_length(), __len__(), etc.
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(layers):
        cache.update(k, v, layer_idx=layer_idx)
    return cache


class BToksTokenInjector(nn.Module):
    """Inject bottleneck btoks tokens into input sequences.

    This module appends btoks token IDs to input_ids and extends
    attention_mask accordingly. It also outputs a btoks_token_mask
    for downstream BToksPooling.

    The module has no trainable parameters. Token embeddings are trained
    via PEFT trainable_token_indices on the backbone's embedding layer.

    Behavior modes (driven by features dict keys):
    - Embed mode (default): append btoks tokens, output btoks_token_mask
    - Generation mode (past_key_values present): skip injection entirely

    Args:
        num_tokens: Number of btoks tokens to inject.

    Example:
        >>> injector = BToksTokenInjector(num_tokens=16)
        >>> # After prepare():
        >>> output = injector(input_ids=ids, attention_mask=mask)
        >>> output["btoks_token_mask"]  # (B, L+num_tokens) bool mask
    """

    def __init__(self, num_tokens: int = 16, **kwargs):
        """Initialize BToksTokenInjector.

        Args:
            num_tokens: Number of btoks tokens to inject.
        """
        super().__init__()
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        self.num_tokens = num_tokens
        # Populated by prepare() — list of token IDs
        self._token_ids: list[int] | None = None

    @classmethod
    def from_config(cls, config: dict) -> BToksTokenInjector:
        """Create a token injector from a module configuration dictionary."""
        config = config.copy()
        config.pop("type", None)
        return cls(**config)

    def on_processor(self, model: Any, processor: Any) -> None:
        """Register btoks tokens in tokenizer and resize backbone embeddings.

        This method is idempotent: calling it multiple times with the same
        tokenizer produces the same result.

        Called automatically by the public model setup path.

        Args:
            model: Retrieval model instance used to find the backbone for resize.
            processor: HuggingFace processor (or tokenizer directly).
        """
        if self.num_tokens == 0:
            logger.info("BToksTokenInjector: num_tokens=0, skipping tokenizer setup")
            self._token_ids = None
            return

        # Resolve tokenizer from processor if needed
        tok = getattr(processor, "tokenizer", processor)

        # Generate token names
        token_names = [_btoks_token_name(i) for i in range(self.num_tokens)]

        # Add tokens that don't already exist (idempotent)
        existing = set(tok.get_vocab().keys())
        new_tokens = [t for t in token_names if t not in existing]
        if new_tokens:
            num_added = tok.add_tokens(new_tokens, special_tokens=True)
            logger.info(f"BToksTokenInjector: added {num_added} btoks tokens to tokenizer")

        # Cache token IDs
        self._token_ids = tok.convert_tokens_to_ids(token_names)
        logger.info(f"BToksTokenInjector: token_ids={self._token_ids}")

        # Find backbone and resize embeddings, init new tokens from EOS
        backbone = model.find_backbone()
        if backbone is not None:
            vocab_size = len(tok)
            embed = backbone.model.get_input_embeddings()
            current_size = embed.weight.shape[0]
            if vocab_size > current_size:
                # Snapshot EOS embedding before resize
                eos_id = tok.eos_token_id
                eos_vec = embed.weight.data[eos_id].clone()

                backbone.model.resize_token_embeddings(vocab_size)

                # Initialize new btoks token embeddings with EOS vector
                embed = backbone.model.get_input_embeddings()
                with torch.no_grad():
                    for tid in self._token_ids or []:
                        embed.weight.data[tid] = eos_vec
                logger.info(
                    f"BToksTokenInjector: resized embeddings {current_size} -> {vocab_size}, "
                    f"initialized from eos_token_id={eos_id}"
                )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Append btoks tokens to input sequence.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)
            **kwargs: Passthrough arguments

        Returns:
            Dict with updated input_ids, attention_mask, and btoks_token_mask.
        """
        if self.num_tokens == 0:
            batch_size, seq_len = input_ids.shape
            btoks_token_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=input_ids.device,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "btoks_token_mask": btoks_token_mask,
                **kwargs,
            }

        if self._token_ids is None:
            raise RuntimeError(
                "BToksTokenInjector has not been set up. "
                "Call model.setup(processor=processor) before forward."
            )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Build btoks token IDs tensor: (B, num_tokens)
        btoks_ids = (
            torch.tensor(self._token_ids, dtype=input_ids.dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Truncate front of input if appending would exceed max_length
        # (max_length is not enforced here — backbone handles position limits)
        # Append btoks tokens to input_ids
        new_input_ids = torch.cat([input_ids, btoks_ids], dim=1)

        # Extend attention_mask
        if attention_mask is not None:
            btoks_mask = torch.ones(
                batch_size, self.num_tokens, dtype=attention_mask.dtype, device=device
            )
            new_attention_mask = torch.cat([attention_mask, btoks_mask], dim=1)
        else:
            new_attention_mask = None

        # Build btoks_token_mask: True at btoks token positions
        new_seq_len = new_input_ids.shape[1]
        btoks_token_mask = torch.zeros(batch_size, new_seq_len, dtype=torch.bool, device=device)
        btoks_token_mask[:, seq_len:] = True

        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "btoks_token_mask": btoks_token_mask,
            **kwargs,
        }

    def get_token_ids(self) -> list[int] | None:
        """Return cached btoks token IDs (None if prepare() not called)."""
        return self._token_ids

    def get_trainable_token_indices(self) -> list[int] | None:
        """Return token indices that should be trainable via PEFT.

        This is a generic protocol method called by VLM2Emb.get_trainable_token_indices().
        Any module that injects trainable tokens should implement this method.

        Returns:
            List of token IDs, or None if prepare() not called.
        """
        return self._token_ids


class BToksPooling(nn.Module):
    """Pool btoks token hidden states into a single embedding.

    Extracts hidden states at btoks token positions (identified by
    btoks_token_mask) and mean-pools them into a fixed-size embedding.

    In generation mode (loss present in features), skip pooling entirely.

    Example:
        >>> pooling = BToksPooling()
        >>> output = pooling(
        ...     last_hidden_state=hidden,
        ...     btoks_token_mask=mask,
        ... )
        >>> embeddings = output["embeddings"]  # (B, D)
    """

    def __init__(self, **kwargs):
        """Initialize BToksPooling."""
        super().__init__()

    @classmethod
    def from_config(cls, config: dict) -> BToksPooling:
        """Create a pooling module from a module configuration dictionary."""
        config = config.copy()
        config.pop("type", None)
        return cls(**config)

    def forward(
        self,
        last_hidden_state: Tensor,
        btoks_token_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Pool btoks token hidden states.

        Args:
            last_hidden_state: Hidden states (B, L, D)
            btoks_token_mask: Boolean mask (B, L), True at btoks positions
            attention_mask: Attention mask (B, L)
            past_key_values: KV cache (passthrough)
            **kwargs: Passthrough arguments

        Returns:
            Dict with 'embeddings' key containing (B, D) pooled embedding.
        """
        if (
            btoks_token_mask is None
            or btoks_token_mask.numel() == 0
            or not bool(btoks_token_mask.any().item())
        ):
            pooled = pool_last_valid_token(last_hidden_state, attention_mask)
            return {
                "embeddings": pooled,
                "last_hidden_state": last_hidden_state,
                "btoks_token_mask": btoks_token_mask,
                "attention_mask": attention_mask,
                **kwargs,
            }

        # Extract and mean-pool btoks token hidden states
        # btoks_token_mask: (B, L) bool
        mask_expanded = btoks_token_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)  # (B, D)
        count = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = sum_hidden / count  # (B, D)

        return {
            "embeddings": pooled,
            "last_hidden_state": last_hidden_state,
            "btoks_token_mask": btoks_token_mask,
            "attention_mask": attention_mask,
            **kwargs,
        }


class BToksAttentionPooling(nn.Module):
    """Attention-based pooling for btoks tokens with MLP.

    Includes trainable attention weights, a two-layer feed-forward block, and
    a LayerNorm residual connection. This module is used when the config wants
    learned aggregation over multiple BToks token states instead of mean pooling.

    Architecture:
        BToks Tokens Hidden (B, N, D)
            -> Attention (Linear D->1)
            -> Softmax
            -> Weighted Sum
            -> MLP (D -> D*mlp_ratio -> D)
            -> LayerNorm + Residual
            -> Embedding (B, D)

    Args:
        hidden_dim: Hidden dimension of the model (D)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout probability (default: 0.1)
        attention_scale: Whether to scale attention scores (default: True)

    Example:
        >>> pooling = BToksAttentionPooling(hidden_dim=2048, mlp_ratio=4.0)
        >>> output = pooling(
        ...     last_hidden_state=hidden,  # (B, L, D)
        ...     btoks_token_mask=mask,     # (B, L) bool
        ... )
        >>> embeddings = output["embeddings"]  # (B, D)
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attention_scale: bool = True,
    ):
        """Initialize BToksAttentionPooling.

        Args:
            hidden_dim: Hidden dimension of the model (required).
            mlp_ratio: MLP hidden dimension ratio (mlp_hidden = hidden_dim * mlp_ratio)
            dropout: Dropout probability after MLP
            attention_scale: Whether to scale attention scores by sqrt(D)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.attention_scale = attention_scale
        mlp_hidden = int(hidden_dim * mlp_ratio)

        # Attention: compute importance score for each btoks token
        self.attention = nn.Linear(hidden_dim, 1)

        # MLP: two-layer feed-forward network with GELU
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    @classmethod
    def from_config(cls, config: dict) -> BToksAttentionPooling:
        """Create from configuration dict.

        Args:
            config: Configuration dictionary with keys:
                - hidden_dim: Hidden dimension (required)
                - mlp_ratio: MLP ratio (optional, default: 4.0)
                - dropout: Dropout probability (optional, default: 0.1)
                - attention_scale: Scale attention scores (optional, default: True)

        Returns:
            BToksAttentionPooling instance
        """
        config = config.copy()
        config.pop("type", None)
        return cls(**config)

    def forward(
        self,
        last_hidden_state: Tensor | None = None,
        btoks_token_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Pool btoks token hidden states using attention + MLP.

        Args:
            last_hidden_state: Hidden states from backbone (B, L, D)
            btoks_token_mask: Boolean mask (B, L), True at btoks token positions
            attention_mask: Attention mask (B, L) - passthrough
            past_key_values: KV cache - passthrough
            **kwargs: Additional arguments

        Returns:
            Dict with keys:
                - embeddings: Pooled embedding (B, D)
                - last_hidden_state: Original hidden states (B, L, D)
                - btoks_token_mask: Original mask (B, L)
                - attention_mask: Passthrough (B, L)
                - past_key_values: Passthrough
        """
        if last_hidden_state is None:
            raise ValueError("last_hidden_state is required")
        if (
            btoks_token_mask is None
            or btoks_token_mask.numel() == 0
            or not bool(btoks_token_mask.any().item())
        ):
            embeddings = pool_last_valid_token(last_hidden_state, attention_mask)
            return {
                "embeddings": embeddings,
                "last_hidden_state": last_hidden_state,
                "btoks_token_mask": btoks_token_mask,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                **kwargs,
            }

        # Extract btoks token hidden states directly: (B, N, D)
        # BToks tokens are contiguous at the end of sequence (appended by BToksTokenInjector)
        num_btoks = btoks_token_mask.sum(dim=1)[0].item()
        btoks_hidden = last_hidden_state[:, -num_btoks:]  # (B, N, D)

        # ========== Attention ==========
        # Compute attention scores only over btoks tokens
        attn_scores = self.attention(btoks_hidden)  # (B, N, 1)

        # Apply scale factor
        if self.attention_scale:
            attn_scores = attn_scores / (self.hidden_dim**0.5)

        # Softmax over btoks token dimension
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N, 1)

        # Weighted sum
        attended = (btoks_hidden * attn_weights).sum(dim=1)  # (B, D)

        # ========== MLP + Residual ==========
        mlp_out = self.mlp(attended)  # (B, D)
        embeddings = self.layer_norm(attended + mlp_out)  # Residual connection

        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
            "btoks_token_mask": btoks_token_mask,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            **kwargs,
        }

    def get_config(self) -> dict:
        """Get module configuration for serialization.

        Returns:
            Dict with configuration keys
        """
        # Get dropout from the first Dropout layer in MLP (index 2)
        dropout = self.mlp[2].p if hasattr(self, 'mlp') else 0.1

        return {
            "type": "BToksAttentionPooling",
            "hidden_dim": self.hidden_dim,
            "mlp_ratio": self.mlp_ratio,
            "dropout": dropout,
            "attention_scale": self.attention_scale,
        }

    def modules_to_save(self) -> list[str]:
        """Return sub-module names for PEFT ModulesToSaveWrapper.

        These sub-modules (nn.Linear, nn.Sequential, nn.LayerNorm) have
        forward(x) signatures naturally compatible with ModulesToSaveWrapper.

        Returns:
            List of sub-module attribute names.
        """
        return ["attention", "mlp", "layer_norm"]


__all__ = [
    "extract_btoks_kv",
    "extract_kv_at_positions",
    "extract_last_token_kv",
    "get_vision_token_ids",
    "last_valid_token_indices",
    "pool_last_valid_token",
    "BToksTokenInjector",
    "BToksPooling",
    "BToksAttentionPooling",
    "BToks_TOKEN_PREFIX",
    "BToks_TOKEN_SUFFIX",
]
