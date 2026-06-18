"""Collators for batching data.


This module provides collators for preparing batches of multimodal data:
- TrainingCollator: For training with query-positive pairs (ProcessorWrapper-based)
- EvalCollator: For evaluation with distributed gather support

- TrainingCollator query-positive 
- EvalCollator gather 
"""

from vlm2emb.auto import AutoCollator

from .eval_collator import EvalCollator
from .training_collator import TrainingCollator

__all__ = [
    "AutoCollator",
    "TrainingCollator",
    "EvalCollator",
]
