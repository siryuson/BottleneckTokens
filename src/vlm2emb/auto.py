"""Auto classes and predefined registry instances for all component types.

This module provides transformers-style Auto classes for automatic
component discovery and instantiation. Each registry owns one public
component namespace and is used by configuration-driven construction.

Example:
    >>> from vlm2emb import AutoModule, AutoTrainer
    >>>
    >>> # Register with decorator
    >>> @AutoModule.register("Qwen2VLBackbone")
    ... class Qwen2VLBackbone:
    ...     pass
    >>>
    >>> # Create from config
    >>> backbone = AutoModule.from_config({"type": "Qwen2VLBackbone", ...})
    >>>
    >>> # List available components
    >>> print(AutoModule.list_modules())
"""

from vlm2emb.registry import Registry

# ==================== Model registries ====================

AutoModule = Registry("module")
"""Registry for all model pipeline modules (Backbone, Pooling, Normalize, etc.).

This is the unified registry for all model components including:
- Backbones (e.g., Qwen2VLBackbone, LlavaBackbone)
- Pooling modules (e.g., LastTokenPooling, MeanPooling)
- Transform modules (e.g., Normalize)
"""

AutoModel = Registry("model")
"""Registry for complete model classes (e.g., VLM2Emb and VLM2Vec)."""

# ==================== Training registries ====================

AutoTrainer = Registry("trainer")
"""Registry for trainer implementations (e.g., VLM2VecTrainer and BToksTrainer)."""

AutoTrainingArgs = Registry("training_args")
"""Registry for training argument presets or factories."""

AutoLoss = Registry("loss")
"""Registry for loss functions (e.g., ContrastiveLoss and InfoNCE)."""

AutoOptimizer = Registry("optimizer")
"""Registry for optimizer factories."""

AutoScheduler = Registry("scheduler")
"""Registry for learning-rate scheduler factories."""

# ==================== Data registries ====================

AutoDataset = Registry("dataset")
"""Registry for dataset loaders (e.g., mmeb_train, vidore, and visrag)."""

AutoCollator = Registry("collator")
"""Registry for data collators."""

AutoSampler = Registry("sampler")
"""Registry for data samplers."""

AutoProcessorWrapper = Registry("processor_wrapper")
"""Registry for processor wrappers (e.g., Qwen2VL, LlavaNext, InternVL2).

ProcessorWrappers wrap transformers.AutoProcessor to provide unified input/output interface
for different VLM backends, handling model-specific token routing and processing.
"""

# ==================== Evaluation registries ====================

AutoMetric = Registry("metric")
"""Registry for evaluation metrics."""

AutoEvaluator = Registry("evaluator")
"""Registry for evaluator implementations (e.g., RetrievalEvaluator)."""

AutoBenchmark = Registry("benchmark")
"""Registry for benchmark definitions (e.g., MMEBBenchmark and MTEBBenchmark)."""


# ==================== Utility functions ====================

_REGISTRIES = {
    # Model related
    "module": AutoModule,
    "model": AutoModel,
    # Training related
    "trainer": AutoTrainer,
    "training_args": AutoTrainingArgs,
    "loss": AutoLoss,
    "optimizer": AutoOptimizer,
    "scheduler": AutoScheduler,
    # Data related
    "dataset": AutoDataset,
    "collator": AutoCollator,
    "sampler": AutoSampler,
    "processor_wrapper": AutoProcessorWrapper,
    # Evaluation related
    "metric": AutoMetric,
    "evaluator": AutoEvaluator,
    "benchmark": AutoBenchmark,
}


def get_registry(name: str) -> Registry:
    """Get a predefined registry by name.

    Args:
        name: Registry name (e.g., "backbone", "trainer", "dataset")

    Returns:
        Registry instance

    Raises:
        ValueError: If registry not found

    Example:
        >>> registry = get_registry("backbone")
        >>> print(registry.list_modules())
    """
    if name not in _REGISTRIES:
        raise ValueError(
            f"Registry '{name}' not found. Available: {list(_REGISTRIES.keys())}"
        )
    return _REGISTRIES[name]


def list_registries() -> list[str]:
    """List all available registry names.

    Returns:
        List of registry names
    """
    return list(_REGISTRIES.keys())


__all__ = [
    # Model related
    "AutoModule",
    "AutoModel",
    # Training related
    "AutoTrainer",
    "AutoTrainingArgs",
    "AutoLoss",
    "AutoOptimizer",
    "AutoScheduler",
    # Data related
    "AutoDataset",
    "AutoCollator",
    "AutoSampler",
    "AutoProcessorWrapper",
    # Evaluation related
    "AutoMetric",
    "AutoEvaluator",
    "AutoBenchmark",
    # Utilities
    "get_registry",
    "list_registries",
]
