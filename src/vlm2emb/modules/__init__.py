"""Composable modules for VLM2Emb/BToks embedding pipelines.

This package contains modules that can be composed into a pipeline for
embedding extraction. Each module receives keyword arguments from the
features dict and returns a dict that is merged back into features.

Module types:
    - Backbone: VLM backbone (Qwen2VLBackbone, etc.)
    - Pooling: Convert sequences to fixed-size vectors (LastTokenPooling, MeanPooling)
    - Normalize: L2 normalization

Key conventions:
    - last_hidden_state: (B, L, D) - Transformer output
    - attention_mask: (B, L) - Valid token mask
    - embedding: (B, D) - Final embedding

Example:
    >>> from vlm2emb.modules import Qwen2VLBackbone, LastTokenPooling
    >>>
    >>> backbone = Qwen2VLBackbone.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> pooling = LastTokenPooling()
    >>>
    >>> # Pipeline flow
    >>> features = {"input_ids": ..., "pixel_values": ...}
    >>> features.update(backbone(**features))
    >>> features.update(pooling(**features))
    >>> embeddings = features["embeddings"]
"""

from .backbone import BackboneBase, Qwen2_5VLBackbone, Qwen2VLBackbone, Qwen3VLBackbone
from .pooling import LastTokenPooling, MeanPooling, Normalize
from .btoks import BToksAttentionPooling, BToksPooling, BToksTokenInjector


def register_modules():
    """Register all modules to AutoModule.

    This function is called when modules are imported to ensure
    all modules are available in the registry.
    """
    from vlm2emb.auto import AutoModule

    AutoModule.register("Qwen2VLBackbone")(Qwen2VLBackbone)
    AutoModule.register("Qwen2_5VLBackbone")(Qwen2_5VLBackbone)
    AutoModule.register("Qwen3VLBackbone")(Qwen3VLBackbone)
    AutoModule.register("LastTokenPooling")(LastTokenPooling)
    AutoModule.register("MeanPooling")(MeanPooling)
    AutoModule.register("Normalize")(Normalize)
    AutoModule.register("BToksTokenInjector")(BToksTokenInjector)
    AutoModule.register("BToksPooling")(BToksPooling)
    AutoModule.register("BToksAttentionPooling")(BToksAttentionPooling)


# Register modules on import
register_modules()


__all__ = [
    "BackboneBase",
    "Qwen2VLBackbone",
    "Qwen2_5VLBackbone",
    "Qwen3VLBackbone",
    "LastTokenPooling",
    "MeanPooling",
    "Normalize",
    "BToksTokenInjector",
    "BToksPooling",
    "BToksAttentionPooling",
    "register_modules",
]
