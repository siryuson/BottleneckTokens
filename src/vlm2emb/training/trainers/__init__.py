"""Specific trainer implementations.

This module provides specific trainer implementations for different
embedding model architectures.

Available trainers:
    - VLM2VecTrainer: Trainer for VLM2Vec with optional GradCache
    - GMEStyleTrainer: GME-style dense trainer alias for fair comparisons

Available training arguments:
    - VLM2VecTrainingArgs: Training arguments for VLM2Vec
    - GMEStyleTrainingArgs: Training arguments for GME-style baselines
"""

from .gme_trainer import GMEStyleTrainer, GMEStyleTrainingArgs
from .btoks_trainer import BToksTrainer, BToksTrainingArgs
from .vlm2vec_trainer import VLM2VecTrainer, VLM2VecTrainingArgs

__all__ = [
    "GMEStyleTrainer",
    "GMEStyleTrainingArgs",
    "BToksTrainer",
    "BToksTrainingArgs",
    "VLM2VecTrainer",
    "VLM2VecTrainingArgs",
]
