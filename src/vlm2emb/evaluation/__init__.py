"""Evaluation APIs for the BToks public runtime.

Provides benchmarks, evaluators, datasets, and metrics for model evaluation.

Example:
    >>> from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
    >>> from vlm2emb.evaluation import RetrievalEvaluator, RetrievalEvalDataset
    >>> from vlm2emb.evaluation import MMEBBenchmark
    >>>
    >>> # Single dataset evaluation
    >>> dataset = build_mmeb_eval_dataset(
    ...     dataset_name="ImageNet-1K",
    ...     artifact_path="/path/to/ImageNet-1K",
    ... )
    >>> evaluator = RetrievalEvaluator(dataset=dataset, batch_size=32)
    >>> results = evaluator(model, processor)
    >>>
    >>> # Benchmark evaluation
    >>> benchmark = MMEBBenchmark(data_path="/path/to/MMEB-V2")
    >>> results = benchmark(model, processor)
"""

# Re-export from new location for backward compatibility.
from vlm2emb.data.datasets.eval import RetrievalEvalDataset
from vlm2emb.evaluation.benchmarks import BaseBenchmark, MMEBBenchmark
from vlm2emb.evaluation.evaluators import BaseEvaluator, RetrievalEvaluator
from vlm2emb.evaluation.metrics import (
    RankingMetrics,
    compute_classification_metrics,
    compute_retrieval_metrics,
)

__all__ = [
    # Benchmarks
    "BaseBenchmark",
    "MMEBBenchmark",
    # Evaluators
    "BaseEvaluator",
    "RetrievalEvaluator",
    # Datasets
    "RetrievalEvalDataset",
    # Metrics
    "RankingMetrics",
    "compute_retrieval_metrics",
    "compute_classification_metrics",
]
