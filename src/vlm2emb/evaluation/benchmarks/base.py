"""Abstract base class for benchmark collections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from vlm2emb.utils.logging import RankLogger

if TYPE_CHECKING:
    import torch.nn as nn

    from vlm2emb.evaluation.evaluators.base import BaseEvaluator

logger = RankLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for all benchmarks.

    Benchmarks organize multiple evaluators and define aggregation logic.
    Provides a standard evaluation loop with progress logging and summary.

    Args:
        name: Benchmark name
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._evaluators: list[BaseEvaluator] | None = None

    @property
    @abstractmethod
    def evaluators(self) -> list[BaseEvaluator]:
        """Get list of evaluators."""
        ...

    @abstractmethod
    def aggregate(self, results: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Aggregate results from all evaluators.

        Args:
            results: Dict mapping dataset name to metric scores

        Returns:
            Aggregated summary dict
        """
        ...

    def __call__(
        self,
        model: nn.Module,
        processor_wrapper: Any,
        accelerator: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run all evaluators with progress logging, aggregate, and print summary.

        Args:
            model: Model to evaluate
            processor_wrapper: ProcessorWrapper for data preprocessing
            accelerator: Optional accelerator for distributed evaluation
            **kwargs: Additional arguments passed to evaluators

        Returns:
            Dict with per-dataset results and "_summary" key. Distributed
            evaluators are expected to return identical scores on every rank,
            so benchmark aggregation produces identical results on every rank.
        """
        results: dict[str, Any] = {}
        total = len(self.evaluators)

        logger.info(f"Starting evaluation: {total} datasets")

        for i, evaluator in enumerate(self.evaluators, 1):
            logger.info(f"[{i}/{total}] Evaluating {evaluator.name}...")
            scores = evaluator(model, processor_wrapper, accelerator, **kwargs)
            results[evaluator.name] = scores

        is_main_process = accelerator is None or bool(accelerator.is_main_process)

        summary = self.aggregate(results)
        results["_summary"] = summary

        # Keep user-visible summaries on the main process only.
        parts = []
        if is_main_process:
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
        if is_main_process and parts:
            logger.info(f"Summary: {', '.join(parts)}")

        return results

    def __len__(self) -> int:
        """Return number of evaluators."""
        return len(self.evaluators)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, num_evaluators={len(self)})"


__all__ = ["BaseBenchmark"]
