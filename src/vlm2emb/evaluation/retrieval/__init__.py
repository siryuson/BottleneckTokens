"""Retrieval evaluation module (backward compatibility).

This module re-exports from the new locations for backward compatibility.
"""

# Re-export from new locations
from vlm2emb.data.datasets.eval import RetrievalEvalDataset
from vlm2emb.evaluation.evaluators.retrieval import RetrievalEvaluator

__all__ = [
    "RetrievalEvalDataset",
    "RetrievalEvaluator",
]
