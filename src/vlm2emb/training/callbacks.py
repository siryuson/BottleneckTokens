"""Training callback system for extensible training hooks.


This module provides a callback base class with fixed hook points
that can be extended to add custom behavior during training.

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from vlm2emb.training.utils.metrics import MetricsAccumulator

if TYPE_CHECKING:
    BaseTrainer = Any


logger = logging.getLogger(__name__)


class Callback:
    """Base class for training callbacks.


    Callbacks allow customizing the training loop behavior by hooking into
    specific points during training. All hook methods use **kwargs for
    future extensibility.

     **kwargs 

    Example:
        >>> class MyCallback(Callback):
        ...     def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        ...         if metrics.get("loss") < self.best_loss:
        ...             trainer.save_checkpoint("best")
        ...
        >>> trainer = BaseTrainer(model, callbacks=[MyCallback()])
        >>> trainer.train()
    """

    def on_train_begin(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Called at the start of training.


        Args:
            trainer: The trainer instance.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    def on_train_end(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Called at the end of training.


        Args:
            trainer: The trainer instance.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    def on_epoch_begin(
        self, trainer: BaseTrainer, epoch: int, **kwargs: Any
    ) -> None:
        """Called at the start of each epoch.

         epoch 

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number (0-indexed).
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    def on_epoch_end(
        self, trainer: BaseTrainer, epoch: int, metrics: dict[str, Any], **kwargs: Any
    ) -> None:
        """Called at the end of each epoch.

         epoch 

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    def on_step_begin(
        self, trainer: BaseTrainer, step: int, **kwargs: Any
    ) -> None:
        """Called before each training step.


        Args:
            trainer: The trainer instance.
            step: Current global step number.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    def on_step_end(
        self,
        trainer: BaseTrainer,
        step: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Called after each training step.


        Args:
            trainer: The trainer instance.
            step: Current global step number.
            loss: Loss value for this step.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    def on_evaluate(
        self,
        trainer: BaseTrainer,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called after validation/evaluation.


        Args:
            trainer: The trainer instance.
            metrics: Dictionary of validation metrics.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass


class LoggingCallback(Callback):
    """Built-in callback for logging training progress.


    Logs training events at configurable intervals with distributed-aware
    logging that only logs from the main process. Uses MetricsAccumulator
    for proper throughput calculation and metrics aggregation.

     MetricsAccumulator 

    Args:
        log_interval: Log every N steps.
        log_epoch_metrics: Whether to log epoch metrics.
        show_throughput: Whether to show throughput in logs.
        batch_size: Batch size for throughput calculation.

    Example:
        >>> callback = LoggingCallback(log_interval=10, show_throughput=True, batch_size=32)
        >>> trainer = BaseTrainer(model, callbacks=[callback])
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_epoch_metrics: bool = True,
        show_throughput: bool = False,
        batch_size: int = 1,
    ):
        """Initialize logging callback.


        Args:
            log_interval: Log every N steps.
            log_epoch_metrics: Whether to log epoch metrics.
            show_throughput: Whether to show throughput.
            batch_size: Batch size for throughput calculation.
        """
        self.log_interval = log_interval
        self.log_epoch_metrics = log_epoch_metrics
        self.show_throughput = show_throughput
        self.batch_size = batch_size
        self._step_losses: list[float] = []
        self._epoch_start_time: float = 0.0
        self._train_start_time: float = 0.0
        self._step_start_time: float = 0.0
        # Use MetricsAccumulator for interval-level loss and throughput metrics.
        self._metrics_accumulator = MetricsAccumulator(batch_size=batch_size)

    def on_train_begin(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Log training start.

        """
        self._train_start_time = time.time()
        if trainer.accelerator.is_main_process:
            logger.info(
                f"Starting training: max_epochs={trainer.max_epochs}, "
                f"gradient_accumulation_steps={trainer.gradient_accumulation_steps}"
            )

    def on_train_end(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Log training completion.

        """
        if trainer.accelerator.is_main_process:
            total_time = time.time() - self._train_start_time
            logger.info(f"Training completed in {total_time:.2f}s")

    def on_epoch_begin(
        self, trainer: BaseTrainer, epoch: int, **kwargs: Any
    ) -> None:
        """Log epoch start and reset epoch-local metric state."""
        self._step_losses = []
        self._epoch_start_time = time.time()
        self._metrics_accumulator.reset()
        if trainer.accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch + 1}/{trainer.max_epochs}")

    def on_epoch_end(
        self, trainer: BaseTrainer, epoch: int, metrics: dict[str, Any], **kwargs: Any
    ) -> None:
        """Log epoch-level metrics on the main process."""
        if not self.log_epoch_metrics or not trainer.accelerator.is_main_process:
            return

        epoch_time = time.time() - self._epoch_start_time
        avg_loss = sum(self._step_losses) / len(self._step_losses) if self._step_losses else 0.0

        log_msg = f"Epoch {epoch + 1} completed in {epoch_time:.2f}s, avg_loss={avg_loss:.4f}"
        for key, value in metrics.items():
            if isinstance(value, float):
                log_msg += f", {key}={value:.4f}"
            else:
                log_msg += f", {key}={value}"
        logger.info(log_msg)

    def on_step_begin(
        self, trainer: BaseTrainer, step: int, **kwargs: Any
    ) -> None:
        """Track step start time for throughput calculation.

        """
        self._step_start_time = time.time()

    def on_step_end(
        self,
        trainer: BaseTrainer,
        step: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Log step progress with optional throughput.

        """
        self._step_losses.append(loss)

        # Accumulate metrics using MetricsAccumulator.
        step_time = time.time() - self._step_start_time
        self._metrics_accumulator.add(loss=loss, step_time=step_time)

        if not trainer.accelerator.is_main_process:
            return

        if step % self.log_interval == 0 and step > 0:
            # Calculate interval metrics using MetricsAccumulator.
            interval_metrics = self._metrics_accumulator.compute()
            avg_loss = interval_metrics["avg_loss"]

            log_msg = f"Step {step}: loss={avg_loss:.4f}"

            if self.show_throughput:
                throughput = interval_metrics["throughput"]
                log_msg += f", throughput={throughput:.1f} samples/sec"

            logger.info(log_msg)
            self._metrics_accumulator.reset()


class ProgressCallback(Callback):
    """Callback for tracking training progress with ETA.


    Args:
        total_steps: Total number of training steps (optional).

    Example:
        >>> callback = ProgressCallback()
        >>> trainer = BaseTrainer(model, callbacks=[callback])
    """

    def __init__(self, total_steps: int | None = None):
        """Initialize progress callback.

        """
        self.total_steps = total_steps
        self._start_time: float = 0.0
        self._step_times: list[float] = []

    def on_train_begin(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Initialize progress tracking.

        """
        self._start_time = time.time()
        self._step_times = []

    def on_step_end(
        self,
        trainer: BaseTrainer,
        step: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Update progress tracking.

        """
        self._step_times.append(time.time())

        if not trainer.accelerator.is_main_process:
            return

        if len(self._step_times) >= 2:
            # Calculate average step time
            avg_step_time = (self._step_times[-1] - self._start_time) / len(self._step_times)

            if self.total_steps is not None:
                remaining_steps = self.total_steps - step
                eta_seconds = remaining_steps * avg_step_time
                eta_formatted = self._format_time(eta_seconds)
                progress = step / self.total_steps * 100
                logger.debug(
                    f"Progress: {progress:.1f}% ({step}/{self.total_steps}), "
                    f"ETA: {eta_formatted}"
                )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time.

        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class ValidationCallback(Callback):
    """Callback for triggering validation at specified intervals.


    This callback can trigger validation based on step count or epoch end.
    It works with the Trainer's built-in _validate() method and integrates
    with CheckpointManager for best checkpoint saving.

     epoch 
     Trainer  _validate() 
    CheckpointManager 

    Args:
        eval_steps: Run validation every N steps. None to disable.
        eval_on_epoch_end: Run validation at epoch end.
        metric_for_best: Metric name to track for best model.
        greater_is_better: Whether higher metric is better.
        checkpoint_manager: CheckpointManager for saving best checkpoint.

    Example:
        >>> from vlm2emb.training.checkpoint import CheckpointManager
        >>> ckpt_manager = CheckpointManager("checkpoints", keep_checkpoints=3)
        >>> callback = ValidationCallback(
        ...     eval_steps=500,
        ...     metric_for_best="eval_loss",
        ...     greater_is_better=False,
        ...     checkpoint_manager=ckpt_manager
        ... )
        >>> trainer = BaseTrainer(model, callbacks=[callback])
    """

    def __init__(
        self,
        eval_steps: int | None = None,
        eval_on_epoch_end: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        checkpoint_manager: Any = None,
    ):
        """Initialize validation callback.


        Args:
            eval_steps: Validate every N steps. None to disable.
            eval_on_epoch_end: Validate at epoch end.
            metric_for_best: Metric for best model tracking.
            greater_is_better: Whether higher is better.
            checkpoint_manager: Manager for saving best checkpoint.
        """
        self.eval_steps = eval_steps
        self.eval_on_epoch_end = eval_on_epoch_end
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.checkpoint_manager = checkpoint_manager
        self._best_metric: float | None = None

    def on_step_end(
        self,
        trainer: BaseTrainer,
        step: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Trigger validation at step intervals if configured.

        """
        if self.eval_steps is None:
            return

        if step > 0 and step % self.eval_steps == 0:
            # Run validation
            if trainer.val_loader is not None:
                metrics = trainer._validate()
                self._maybe_save_best(trainer, metrics)

    def on_epoch_end(
        self,
        trainer: BaseTrainer,
        epoch: int,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Trigger validation at epoch end if configured."""
        if not self.eval_on_epoch_end:
            return

        # Check whether trainer-level validation already added val_* metrics.
        if any(k.startswith("val_") for k in metrics):
            # Validation already ran, just check for best
            self._maybe_save_best(trainer, metrics)
        elif trainer.val_loader is not None:
            # Run validation if not already done
            val_metrics = trainer._validate()
            self._maybe_save_best(trainer, val_metrics)

    def on_evaluate(
        self,
        trainer: BaseTrainer,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle validation metrics for best checkpoint tracking.

        """
        # Log validation results
        if trainer.accelerator.is_main_process:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            logger.info(f"Validation: {metric_str}")

    def _maybe_save_best(self, trainer: BaseTrainer, metrics: dict[str, Any]) -> None:
        """Save best checkpoint if metric improved.

        """
        # Get metric value (handle both "eval_loss" and "val_loss" naming)
        metric_value = metrics.get(self.metric_for_best)
        if metric_value is None:
            # Try with val_ prefix
            alt_name = f"val_{self.metric_for_best.replace('eval_', '')}"
            metric_value = metrics.get(alt_name)

        if metric_value is None:
            return

        # Check if this is a new best
        is_better = False
        if self._best_metric is None:
            is_better = True
        elif self.greater_is_better and metric_value > self._best_metric:
            is_better = True
        elif not self.greater_is_better and metric_value < self._best_metric:
            is_better = True

        if is_better:
            self._best_metric = metric_value
            if trainer.accelerator.is_main_process:
                logger.info(
                    f"New best {self.metric_for_best}: {metric_value:.4f}"
                )

            # Save best checkpoint if manager available
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_best(
                    metric_name=self.metric_for_best,
                    metric_value=metric_value,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    trainer_state={
                        "step": trainer.global_step,
                        "epoch": trainer.current_epoch,
                    },
                    higher_better=self.greater_is_better,
                )


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric stops improving.


    This callback monitors a specified metric and stops training
    if no improvement is seen for `patience` evaluations.

     `patience` 

    Args:
        metric_name: Name of the metric to monitor.
        patience: Number of evaluations without improvement before stopping.
        greater_is_better: Whether higher metric values are better.
        min_delta: Minimum change to qualify as an improvement.

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     metric_name="eval_loss",
        ...     patience=3,
        ...     greater_is_better=False,
        ...     min_delta=0.001
        ... )
        >>> trainer = BaseTrainer(model, callbacks=[callback])
    """

    def __init__(
        self,
        metric_name: str = "eval_loss",
        patience: int = 3,
        greater_is_better: bool = False,
        min_delta: float = 0.0,
    ):
        """Initialize early stopping callback.


        Args:
            metric_name: Metric to monitor.
            patience: Evaluations without improvement before stopping.
            greater_is_better: Whether higher is better.
            min_delta: Minimum change for improvement.
        """
        self.metric_name = metric_name
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.min_delta = min_delta

        self._best_value: float | None = None
        self._evaluations_without_improvement = 0

    def on_evaluate(
        self,
        trainer: BaseTrainer,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Check for improvement and trigger early stopping if needed.

        """
        # Get metric value (handle both direct name and val_ prefix)
        current = metrics.get(self.metric_name)
        if current is None:
            # Try with val_ prefix
            alt_name = f"val_{self.metric_name.replace('eval_', '')}"
            current = metrics.get(alt_name)

        if current is None:
            return

        # First evaluation
        if self._best_value is None:
            self._best_value = current
            return

        # Check for improvement
        if self.greater_is_better:
            is_improvement = current > self._best_value + self.min_delta
        else:
            is_improvement = current < self._best_value - self.min_delta

        if is_improvement:
            self._best_value = current
            self._evaluations_without_improvement = 0
        else:
            self._evaluations_without_improvement += 1

        # Check if should stop
        if self._evaluations_without_improvement >= self.patience:
            if trainer.accelerator.is_main_process:
                logger.info(
                    f"Early stopping triggered: {self.metric_name} has not improved "
                    f"for {self.patience} evaluations. Best: {self._best_value:.4f}"
                )
            trainer.should_stop = True

    def reset(self) -> None:
        """Reset the early stopping state.

        """
        self._best_value = None
        self._evaluations_without_improvement = 0


class CallbackList:
    """Container for managing multiple callbacks.


    This class allows treating multiple callbacks as a single callback,
    dispatching events to all contained callbacks in order.


    Args:
        callbacks: List of callbacks.

    Example:
        >>> callbacks = CallbackList([LoggingCallback(), ProgressCallback()])
        >>> callbacks.on_train_begin(trainer)  # Calls all callbacks
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        """Initialize callback list.

        """
        self.callbacks: list[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback.

        """
        self.callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        """Remove a callback.

        """
        self.callbacks.remove(callback)

    def on_train_begin(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Dispatch on_train_begin to all callbacks.

         on_train_begin 
        """
        for callback in self.callbacks:
            callback.on_train_begin(trainer, **kwargs)

    def on_train_end(self, trainer: BaseTrainer, **kwargs: Any) -> None:
        """Dispatch on_train_end to all callbacks.

         on_train_end 
        """
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)

    def on_epoch_begin(
        self, trainer: BaseTrainer, epoch: int, **kwargs: Any
    ) -> None:
        """Dispatch on_epoch_begin to all callbacks.

         on_epoch_begin 
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch, **kwargs)

    def on_epoch_end(
        self, trainer: BaseTrainer, epoch: int, metrics: dict[str, Any], **kwargs: Any
    ) -> None:
        """Dispatch on_epoch_end to all callbacks.

         on_epoch_end 
        """
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics, **kwargs)

    def on_step_begin(
        self, trainer: BaseTrainer, step: int, **kwargs: Any
    ) -> None:
        """Dispatch on_step_begin to all callbacks.

         on_step_begin 
        """
        for callback in self.callbacks:
            callback.on_step_begin(trainer, step, **kwargs)

    def on_step_end(
        self,
        trainer: BaseTrainer,
        step: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Dispatch on_step_end to all callbacks.

         on_step_end 
        """
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, loss, **kwargs)

    def on_evaluate(
        self,
        trainer: BaseTrainer,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Dispatch on_evaluate to all callbacks.

         on_evaluate 
        """
        for callback in self.callbacks:
            callback.on_evaluate(trainer, metrics, **kwargs)

    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)
