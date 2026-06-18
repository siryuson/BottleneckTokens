"""Training components and orchestration.

This module provides:
- Base Trainer: Trainer (generic embedding model trainer)
- Specific Trainers: VLM2VecTrainer (in trainers/)
- Losses: Contrastive and distributed losses
"""

from vlm2emb.auto import AutoLoss, AutoTrainer

# Import trainers submodule to trigger trainer registration.
from . import trainers  # noqa: F401
from .trainer import Trainer

__all__ = [
    # Registries
    "AutoTrainer",
    "AutoLoss",
    # Base Trainer
    "Trainer",
]
