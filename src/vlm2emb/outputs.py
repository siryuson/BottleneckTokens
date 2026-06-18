"""Output types and protocols for the BToks public runtime.

This module defines:
- Output dataclasses for model components (BackboneOutput, EmbedderOutput, etc.)
- Protocols (interfaces) for type checking (Embedder, Generator, etc.)
- Type aliases for common types

Example:
    >>> from vlm2emb.outputs import EmbedderOutput, Embedder
    >>>
    >>> # Check if model implements Embedder protocol
    >>> if isinstance(model, Embedder):
    ...     output: EmbedderOutput = model.encode(input_ids, pixel_values=images)
    ...     embeddings = output.embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

from torch import Tensor

# ============================================================================
# TypedDict input schemas.
# ============================================================================


class ImageInput(TypedDict, total=False):
    """Input tensor fields accepted for image examples."""

    pixel_values: Tensor  # (B, C, H, W) or (B, N_patches, C, H, W) for dynamic
    image_grid_thw: Tensor | None  # (B, 3) for Qwen2-VL dynamic resolution


class VideoInput(TypedDict, total=False):
    """Input tensor fields accepted for video examples."""

    pixel_values: Tensor  # (B, T, C, H, W) or (B, N_frames, C, H, W)
    video_grid_thw: Tensor | None  # (B, 3) for Qwen2-VL


class TextInput(TypedDict, total=False):
    """Input tensor fields accepted for text examples."""

    input_ids: Tensor  # (B, L)
    attention_mask: Tensor | None  # (B, L)


# ============================================================================
# Output dataclasses.
# ============================================================================


@dataclass
class ModelInput:
    """Unified input format for all modalities.

    Modality type is inferred from the presence of fields:
    - Only text: TEXT modality
    - text + image: IMAGE/MULTIMODAL modality
    - text + video: VIDEO modality
    """

    text: TextInput
    image: ImageInput | None = None
    video: VideoInput | None = None


@dataclass
class BackboneOutput:
    """Output from backbone models.

    Attributes:
        last_hidden_state: Last layer hidden states (B, L, D) - Required
        hidden_states: Tuple of hidden states from all layers
        attention_mask: Attention mask (B, L)
        embedding_head_output: Optional pooled representation (B, D)
        modality_mask: Optional mask indicating modality type per token
        past_key_values: Optional cached key-values for generation
        metadata: Additional outputs
    """

    last_hidden_state: Tensor  # (B, L, D) - Required
    hidden_states: tuple[Tensor, ...] | None = None
    attention_mask: Tensor | None = None
    embedding_head_output: Tensor | None = None
    modality_mask: Tensor | None = None
    past_key_values: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.last_hidden_state.size(0)

    @property
    def sequence_length(self) -> int:
        """Return sequence length."""
        return self.last_hidden_state.size(1)

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension size."""
        return self.last_hidden_state.size(2)


@dataclass
class EmbeddingHeadOutput:
    """Output from embedding head operations.

    Attributes:
        embeddings: Embedding vectors (B, D)
        pooling_mask: Mask indicating which tokens were pooled (B, L)
        strategy: The pooling strategy used
        metadata: Additional information about pooling
    """

    embeddings: Tensor  # (B, D)
    pooling_mask: Tensor | None = None
    strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonerOutput:
    """Output from reasoning module (think-then-embed).

    Attributes:
        reasoning_text: Generated reasoning text
        reasoning_embeddings: Optional embeddings of reasoning tokens (B, L_r, D)
        hidden_states: Hidden states from reasoner (B, L_r, D)
        metadata: Additional outputs
    """

    reasoning_text: str
    reasoning_embeddings: Tensor | None = None
    hidden_states: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedderOutput:
    """Output container for embedding models.

    This is the primary output type for models that encode inputs into
    fixed-size embedding vectors.

    Attributes:
        embeddings: Final embedding vectors (B, D) - Required
        normalized: Whether embeddings are L2-normalized
        last_hidden_state: Optional backbone output (B, L, D)
        attention_mask: Optional attention mask (B, L)
        backbone_output: Optional full backbone output
        embedding_head_output: Optional embedding head output
        reasoner_output: Optional reasoning output (for think-then-embed)
        metadata: Additional model-specific outputs

    Example:
        >>> output = model.encode(input_ids, pixel_values=images)
        >>> embeddings = output.embeddings  # (B, D)
        >>> similarity = embeddings @ embeddings.T  # cosine sim if normalized
    """

    embeddings: Tensor  # (B, D) - Required
    normalized: bool = True
    last_hidden_state: Tensor | None = None
    attention_mask: Tensor | None = None
    backbone_output: BackboneOutput | None = None
    embedding_head_output: EmbeddingHeadOutput | None = None
    reasoner_output: ReasonerOutput | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pooling_output(self) -> EmbeddingHeadOutput | None:
        """Backward compatibility property."""
        return self.embedding_head_output

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.embeddings.size(0)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embeddings.size(1)

    def __repr__(self) -> str:
        return (
            f"EmbedderOutput(shape={tuple(self.embeddings.shape)}, "
            f"normalized={self.normalized})"
        )


@dataclass
class GeneratorOutput:
    """Output container for text generation models.

    Attributes:
        sequences: Generated token IDs (B, L) - Required
        text: Optional list of decoded text strings
        scores: Optional tuple of score tensors per token
        metadata: Additional generation outputs

    Example:
        >>> output = model.generate(input_ids, pixel_values=images)
        >>> if output.text is not None:
        ...     print(output.text[0])
    """

    sequences: Tensor  # (B, L) - Required
    text: list[str] | None = None
    scores: tuple[Tensor, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.sequences.size(0)

    @property
    def sequence_length(self) -> int:
        """Return generated sequence length."""
        return self.sequences.size(1)

    def __repr__(self) -> str:
        has_text = self.text is not None
        return (
            f"GeneratorOutput(shape={tuple(self.sequences.shape)}, "
            f"has_text={has_text})"
        )


# ============================================================================
# Runtime-checkable protocols.
# ============================================================================


@runtime_checkable
class Embedder(Protocol):
    """Protocol for models with embedding capability.

    Any model that can convert multimodal inputs into fixed-size vector
    representations should implement this protocol.

    Example:
        >>> if isinstance(model, Embedder):
        ...     output = model.encode(input_ids, pixel_values=images)
    """

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_sizes: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> EmbedderOutput:
        """Encode multimodal inputs into embedding vectors."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Protocol for models with text generation capability.

    Example:
        >>> if isinstance(model, Generator):
        ...     output = model.generate(input_ids, pixel_values=images)
    """

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_sizes: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Generate text from multimodal inputs."""
        ...


@runtime_checkable
class BackboneProtocol(Protocol):
    """Protocol for backbone models."""

    def forward(self, inputs: ModelInput, **kwargs) -> BackboneOutput:
        """Forward pass through backbone."""
        ...

    def get_hidden_size(self) -> int:
        """Return hidden dimension size."""
        ...


@runtime_checkable
class EmbeddingHeadProtocol(Protocol):
    """Protocol for embedding head modules."""

    def forward(
        self,
        backbone_output: BackboneOutput,
        **kwargs,
    ) -> EmbeddingHeadOutput:
        """Apply embedding head to backbone output."""
        ...


# ============================================================================
# Enumerations and constants.
# ============================================================================


class PoolingStrategy:
    """String constants for embedding pooling strategies used in configs."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    LAST_TOKEN = "last_token"
    FIRST_TOKEN = "first_token"
    CLS_TOKEN = "cls_token"
    BOTTLENECK_MEAN = "bottleneck_mean"
    ATTENTION = "attention"

    @classmethod
    def list_all(cls) -> list[str]:
        """List all available pooling strategies."""
        return [
            cls.MEAN,
            cls.MAX,
            cls.MIN,
            cls.LAST_TOKEN,
            cls.FIRST_TOKEN,
            cls.CLS_TOKEN,
            cls.BOTTLENECK_MEAN,
            cls.ATTENTION,
        ]


# ============================================================================
# Common type aliases.
# ============================================================================

ConfigValue = str | int | float | bool | list | dict | None
ConfigDict = dict[str, ConfigValue]
Batch = dict[str, Tensor]
BatchList = list[Batch]
LossDict = dict[str, Tensor]
MetricDict = dict[str, float]


__all__ = [
    # Input types
    "ImageInput",
    "VideoInput",
    "TextInput",
    "ModelInput",
    # Output types
    "BackboneOutput",
    "EmbeddingHeadOutput",
    "ReasonerOutput",
    "EmbedderOutput",
    "GeneratorOutput",
    # Protocols
    "Embedder",
    "Generator",
    "BackboneProtocol",
    "EmbeddingHeadProtocol",
    # Constants
    "PoolingStrategy",
    # Type aliases
    "ConfigValue",
    "ConfigDict",
    "Batch",
    "BatchList",
    "LossDict",
    "MetricDict",
]
