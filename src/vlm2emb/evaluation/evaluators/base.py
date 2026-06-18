"""Abstract base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn
    from accelerate import Accelerator


class BaseEvaluator(ABC):
    """Base class for all evaluators.

    Evaluators are responsible for:
    - Encoding data using the model
    - Computing task-specific metrics
    - Supporting distributed evaluation via accelerate

    Constructor args are evaluation config (what to evaluate, how to score).
    Runtime resources (model, processor_wrapper, accelerator) are passed to __call__.

    Args:
        name: Evaluator name (used in results dict)
        batch_size: Default batch size for encoding (overridable at call time)
        show_progress: Whether to show progress bar

    Example:
        >>> evaluator = RetrievalEvaluator(dataset=dataset, batch_size=32)
        >>> results = evaluator(model, processor_wrapper, accelerator)
        >>> print(results)
        {"hit@1": 0.85, "ndcg@10": 0.72, ...}
    """

    def __init__(
        self,
        name: str = "",
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        self.name = name
        self.batch_size = batch_size
        self.show_progress = show_progress

    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        processor_wrapper: Any,
        accelerator: Accelerator | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate a model and return metric scores.

        Args:
            model: Model to evaluate
            processor_wrapper: ProcessorWrapper for data preprocessing
            accelerator: Optional accelerator for distributed evaluation
            **kwargs: Additional arguments (e.g., batch_size, num_workers overrides)

        Returns:
            Dictionary of metric scores
        """
        ...

    # ================================================================
    # Distributed helpers (accept accelerator as argument)
    # ================================================================

    @staticmethod
    def _is_distributed(accelerator: Accelerator | None) -> bool:
        """Whether running in distributed mode."""
        if accelerator is None:
            return False
        return accelerator.num_processes > 1

    @staticmethod
    def _is_main_process(accelerator: Accelerator | None) -> bool:
        """Whether this is the main process."""
        if accelerator is None:
            return True
        return accelerator.is_main_process

    @staticmethod
    def _num_processes(accelerator: Accelerator | None) -> int:
        """Number of processes."""
        if accelerator is None:
            return 1
        return accelerator.num_processes

    @staticmethod
    def _process_index(accelerator: Accelerator | None) -> int:
        """Current process index."""
        if accelerator is None:
            return 0
        return accelerator.process_index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, batch_size={self.batch_size})"


__all__ = ["BaseEvaluator"]
