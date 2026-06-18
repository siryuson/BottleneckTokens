"""Benchmarks for VLM2Emb.

Provides benchmark classes for organizing multiple evaluators.
"""

from vlm2emb.evaluation.benchmarks.base import BaseBenchmark
from vlm2emb.evaluation.benchmarks.mmeb import MMEBBenchmark

__all__ = [
    "BaseBenchmark",
    "MMEBBenchmark",
]
