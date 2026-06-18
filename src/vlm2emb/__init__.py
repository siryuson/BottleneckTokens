"""BToks public runtime package.

A modular, registry-driven framework for training and evaluating
vision-language models on images, videos, and documents.

High-level API:
    >>> from vlm2emb import AutoModule
    >>>
    >>> # Register components
    >>> @AutoModule.register("MyBackbone")
    ... class MyBackbone:
    ...     pass
    >>>
    >>> # Create from config
    >>> backbone = AutoModule.from_config({"type": "MyBackbone", ...})
"""

__version__ = "0.1.0"
__author__ = "BToks Authors"

# Import modules to trigger registration.
import vlm2emb.modules  # noqa: F401
from vlm2emb.auto import (
    # Model registries
    AutoCollator,
    # Data registries
    AutoDataset,
    AutoEvaluator,
    AutoLoss,
    # Evaluation registries
    AutoMetric,
    AutoModel,
    AutoModule,
    AutoOptimizer,
    AutoSampler,
    AutoScheduler,
    # Training registries
    AutoTrainer,
    # Utility
    get_registry,
)
from vlm2emb.config import Config, ConfigLoader, load_config

# Evaluation entry point.
from vlm2emb.evaluation.evaluate import evaluate
from vlm2emb.exceptions import (
    ComponentAlreadyExistsError,
    ComponentError,
    ComponentNotFoundError,
    ConfigError,
    ConfigInheritanceError,
    ConfigInterpolationError,
    ConfigNotFoundError,
    ConfigSyntaxError,
    ConfigValidationError,
    RegistryError,
    VLM2EmbError,
)
from vlm2emb.model import (
    BToks,
    BToksConfig,
    VLM2Emb,
    VLM2EmbConfig,
    create_model,
)
from vlm2emb.registry import Registry

# Training entry point.
from vlm2emb.training.train import train

from . import data
from .data import datasets

__all__ = [
    # High-level API
    "create_model",
    "train",
    "evaluate",
    "data",
    "datasets",
    # Model
    "BToks",
    "BToksConfig",
    "VLM2Emb",
    "VLM2EmbConfig",
    "AutoModel",  # Model registry
    "AutoModule",  # Pipeline module registry (unified for all modules)
    # Config
    "Config",
    "ConfigLoader",
    "load_config",
    # Registry
    "Registry",
    "get_registry",
    # Auto registries - Training
    "AutoTrainer",
    "AutoLoss",
    "AutoOptimizer",
    "AutoScheduler",
    # Auto registries - Data
    "AutoDataset",
    "AutoCollator",
    "AutoSampler",
    # Auto registries - Evaluation
    "AutoMetric",
    "AutoEvaluator",
    # Exceptions
    "VLM2EmbError",
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentAlreadyExistsError",
    "ComponentError",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigSyntaxError",
    "ConfigInheritanceError",
    "ConfigValidationError",
    "ConfigInterpolationError",
    # Version
    "__version__",
]
