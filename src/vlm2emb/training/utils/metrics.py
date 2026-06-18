"""Training metrics utilities.

This module provides utilities for accumulating and computing training metrics.
"""

from __future__ import annotations


class MetricsAccumulator:
    """Accumulates and averages training metrics over steps.

    This class is used to track loss and calculate throughput
    over a logging interval, properly handling gradient accumulation.

    Args:
        batch_size: Batch size for throughput calculation.

    Example:
        >>> accumulator = MetricsAccumulator(batch_size=32)
        >>> accumulator.add(loss=0.5, step_time=0.1)
        >>> accumulator.add(loss=0.4, step_time=0.1)
        >>> metrics = accumulator.compute()
        >>> print(f"Avg loss: {metrics['avg_loss']:.4f}")
        >>> print(f"Throughput: {metrics['throughput']:.1f} samples/sec")
    """

    def __init__(self, batch_size: int = 1):
        """Initialize metrics accumulator.

        Args:
            batch_size: Batch size for throughput calculation.
        """
        self.batch_size = batch_size
        self._loss_sum = 0.0
        self._time_sum = 0.0
        self._count = 0

    def add(self, loss: float, step_time: float) -> None:
        """Add a step's loss and wall-clock duration to the accumulator.

        Args:
            loss: Loss value for this step.
            step_time: Time taken for this step in seconds.
        """
        self._loss_sum += loss
        self._time_sum += step_time
        self._count += 1

    def compute(self) -> dict[str, float]:
        """Compute averaged loss and sample throughput for accumulated steps.

        Returns:
            Dictionary with avg_loss and throughput.
        """
        if self._count == 0:
            return {"avg_loss": 0.0, "throughput": 0.0}

        avg_loss = self._loss_sum / self._count
        throughput = (
            (self._count * self.batch_size) / self._time_sum
            if self._time_sum > 0
            else 0.0
        )

        return {
            "avg_loss": avg_loss,
            "throughput": throughput,
            "total_steps": self._count,
            "total_time": self._time_sum,
        }

    def reset(self) -> None:
        """Clear all accumulated counters and timing totals."""
        self._loss_sum = 0.0
        self._time_sum = 0.0
        self._count = 0

    @property
    def count(self) -> int:
        """Return the number of steps currently included in the accumulator."""
        return self._count


__all__ = ["MetricsAccumulator"]
