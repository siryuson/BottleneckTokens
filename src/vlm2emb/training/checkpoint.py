"""Checkpoint management for training state persistence.


This module provides:
- CheckpointManager: Atomic checkpoint saving with rotation
- CheckpointCallback: Callback for automatic checkpoint saving

- CheckpointManager: 
- CheckpointCallback: 
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from accelerate import Accelerator

    BaseTrainer = Any

from vlm2emb.training.callbacks import Callback

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Exception raised for checkpoint-related errors.

    """

    pass


class CheckpointManager:
    """Manages checkpoint saving with atomic writes and rotation.


    This class handles:
    - Atomic checkpoint writing (temp dir + rename)
    - Checkpoint rotation (keep latest N)
    - Best checkpoint tracking
    - Distributed-aware saving (main process only)

    - 

    Args:
        output_dir: Directory for checkpoints.
        keep_checkpoints: Max regular checkpoints to keep (excludes best).
        accelerator: Accelerator for distributed training.

    Example:
        >>> manager = CheckpointManager("checkpoints", keep_checkpoints=3)
        >>> manager.save(step=1000, model=model, optimizer=optimizer, trainer_state={})
        >>> manager.save_best("eval_loss", 0.15, model, optimizer, {})
    """

    def __init__(
        self,
        output_dir: str | Path,
        keep_checkpoints: int = 5,
        accelerator: Accelerator | None = None,
    ):
        """Initialize checkpoint manager.


        Args:
            output_dir: Directory for checkpoints.
            keep_checkpoints: Max checkpoints to keep.
            accelerator: Accelerator for distributed training.
        """
        self.output_dir = Path(output_dir)
        self.keep_checkpoints = keep_checkpoints
        self.accelerator = accelerator
        self._saved_checkpoints: list[Path] = []
        self._best_metric_value: float | None = None
        self._best_metric_name: str | None = None
        self._best_metric_higher_better: bool = False

        # Create output directory
        if self._is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_main_process(self) -> bool:
        """Check if current process is main process.

        """
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process

    def _wait_for_everyone(self) -> None:
        """Synchronize all processes.

        """
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        trainer_state: dict[str, Any],
        epoch: int | None = None,
    ) -> Path | None:
        """Save checkpoint atomically.


        Args:
            step: Current training step.
            model: Model to save.
            optimizer: Optimizer to save.
            trainer_state: Additional trainer state.
            epoch: Current epoch (optional).

        Returns:
            Path to saved checkpoint, or None if not main process.
             None

        Raises:
            CheckpointError: If checkpoint saving fails.
             CheckpointError
        """
        # Only main process saves
        if not self._is_main_process():
            self._wait_for_everyone()
            return None

        checkpoint_dir = self.output_dir / f"step_{step}"
        temp_dir = self.output_dir / f".tmp_step_{step}"

        try:
            # Clean up any existing temp dir
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            temp_dir.mkdir(parents=True, exist_ok=True)

            # Get model state (unwrap if using accelerator)
            if self.accelerator is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                model_state = unwrapped_model.state_dict()
            else:
                model_state = model.state_dict()

            # Save model state
            torch.save(model_state, temp_dir / "model.pt")

            # Save optimizer state
            torch.save(optimizer.state_dict(), temp_dir / "optimizer.pt")

            # Prepare trainer state with metadata
            full_state = {
                "epoch": epoch,
                "step": step,
                "global_step": step,
                "timestamp": datetime.now().isoformat(),
                **trainer_state,
            }

            # Save trainer state
            with open(temp_dir / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump(full_state, f, indent=2, ensure_ascii=False)

            # Atomic rename
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            temp_dir.rename(checkpoint_dir)

            # Verify checkpoint integrity
            self._verify_checkpoint(checkpoint_dir)

            # Track and rotate
            self._saved_checkpoints.append(checkpoint_dir)
            self._rotate_checkpoints()

            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            return checkpoint_dir

        except Exception as e:
            # Clean up temp dir on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise CheckpointError(f"Failed to save checkpoint at step {step}: {e}") from e

        finally:
            self._wait_for_everyone()

    def save_best(
        self,
        metric_name: str,
        metric_value: float,
        model: nn.Module,
        optimizer: Optimizer,
        trainer_state: dict[str, Any],
        higher_better: bool = False,
    ) -> Path | None:
        """Save checkpoint if metric is best so far.


        Args:
            metric_name: Name of the metric.
            metric_value: Current metric value.
            model: Model to save.
            optimizer: Optimizer to save.
            trainer_state: Additional trainer state.
            higher_better: Whether higher metric is better.

        Returns:
            Path to best checkpoint if saved, None otherwise.
             None
        """
        # Check if this is a new best
        is_best = False
        if self._best_metric_value is None:
            is_best = True
        elif higher_better and metric_value > self._best_metric_value:
            is_best = True
        elif not higher_better and metric_value < self._best_metric_value:
            is_best = True

        if not is_best:
            return None

        # Update best tracking
        self._best_metric_value = metric_value
        self._best_metric_name = metric_name
        self._best_metric_higher_better = higher_better

        # Only main process saves
        if not self._is_main_process():
            self._wait_for_everyone()
            return None

        best_dir = self.output_dir / "best"
        temp_dir = self.output_dir / ".tmp_best"

        try:
            # Clean up any existing temp dir
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            temp_dir.mkdir(parents=True, exist_ok=True)

            # Get model state
            if self.accelerator is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                model_state = unwrapped_model.state_dict()
            else:
                model_state = model.state_dict()

            # Save model state
            torch.save(model_state, temp_dir / "model.pt")

            # Save optimizer state
            torch.save(optimizer.state_dict(), temp_dir / "optimizer.pt")

            # Prepare trainer state with best metric info
            full_state = {
                "timestamp": datetime.now().isoformat(),
                "best_metric": {
                    "name": metric_name,
                    "value": metric_value,
                    "higher_better": higher_better,
                },
                **trainer_state,
            }

            # Save trainer state
            with open(temp_dir / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump(full_state, f, indent=2, ensure_ascii=False)

            # Save best metric info separately for easy access
            with open(temp_dir / "best_metric.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": metric_name,
                        "value": metric_value,
                        "higher_better": higher_better,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            # Atomic rename
            if best_dir.exists():
                shutil.rmtree(best_dir)
            temp_dir.rename(best_dir)

            logger.info(
                f"New best checkpoint saved: {metric_name}={metric_value:.4f}"
            )
            return best_dir

        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise CheckpointError(f"Failed to save best checkpoint: {e}") from e

        finally:
            self._wait_for_everyone()

    def _verify_checkpoint(self, checkpoint_dir: Path) -> None:
        """Verify checkpoint integrity.


        Args:
            checkpoint_dir: Checkpoint directory to verify.

        Raises:
            CheckpointError: If verification fails.
             CheckpointError
        """
        required_files = ["model.pt", "optimizer.pt", "trainer_state.json"]
        for filename in required_files:
            filepath = checkpoint_dir / filename
            if not filepath.exists():
                raise CheckpointError(f"Missing checkpoint file: {filepath}")
            if filepath.stat().st_size == 0:
                raise CheckpointError(f"Empty checkpoint file: {filepath}")

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond limit.


        Note: Never deletes the 'best/' checkpoint.
         'best
        """
        while len(self._saved_checkpoints) > self.keep_checkpoints:
            oldest = self._saved_checkpoints.pop(0)
            # Never delete best checkpoint
            if oldest.name == "best":
                continue
            if oldest.exists():
                shutil.rmtree(oldest)
                logger.debug(f"Rotated out old checkpoint: {oldest}")

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint.


        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
             None
        """
        if not self._saved_checkpoints:
            # Try to find existing checkpoints
            step_dirs = sorted(
                self.output_dir.glob("step_*"),
                key=lambda p: int(p.name.split("_")[1]),
            )
            if step_dirs:
                return step_dirs[-1]
            return None
        return self._saved_checkpoints[-1]

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint.


        Returns:
            Path to best checkpoint, or None if no best checkpoint exists.
             None
        """
        best_dir = self.output_dir / "best"
        return best_dir if best_dir.exists() else None

    def find_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint by step number.


        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
             None
        """
        if not self.output_dir.exists():
            return None

        checkpoints = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith("step_"):
                try:
                    step = int(path.name.split("_")[1])
                    checkpoints.append((step, path))
                except (IndexError, ValueError):
                    continue

        if not checkpoints:
            return None

        # Sort by step number and return latest
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint and restore state.


        Args:
            checkpoint_path: Path to checkpoint directory.
            model: Model to load state into.
            optimizer: Optimizer to load state into (optional).
            strict: Whether to strictly enforce state dict key matching.

        Returns:
            Trainer state dictionary with epoch, step, metrics, etc.
             epochstepmetrics 

        Raises:
            CheckpointError: If checkpoint is invalid or corrupted.
             CheckpointError

        Example:
            >>> state = manager.load("checkpoints/step_1000", model, optimizer)
            >>> print(f"Resuming from step {state['step']}")
        """
        checkpoint_dir = Path(checkpoint_path)

        # Validate checkpoint
        self.validate_checkpoint(checkpoint_dir)

        # Load model state
        model_path = checkpoint_dir / "model.pt"
        try:
            model_state = torch.load(model_path, map_location="cpu", weights_only=True)
            if self.accelerator is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(model_state, strict=strict)
            else:
                model.load_state_dict(model_state, strict=strict)
            logger.info(f"Loaded model state from {model_path}")
        except Exception as e:
            raise CheckpointError(f"Failed to load model state from {model_path}: {e}") from e

        # Load optimizer state if provided
        if optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            try:
                optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(optimizer_state)
                logger.info(f"Loaded optimizer state from {optimizer_path}")
            except Exception as e:
                raise CheckpointError(f"Failed to load optimizer state from {optimizer_path}: {e}") from e

        # Load trainer state
        state_path = checkpoint_dir / "trainer_state.json"
        try:
            with open(state_path, encoding="utf-8") as f:
                trainer_state = json.load(f)
        except Exception as e:
            raise CheckpointError(f"Failed to load trainer state from {state_path}: {e}") from e

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
        return trainer_state

    def load_model_only(
        self,
        model_path: str | Path,
        model: nn.Module,
        strict: bool = True,
    ) -> dict[str, set[str]]:
        """Load only model weights without optimizer or trainer state.


        This is useful for:
        - Loading checkpoints for inference
        - Fine-tuning from pretrained weights
        - Partial model loading with strict=False

        -  strict=False 

        Args:
            model_path: Path to model.pt file or checkpoint directory.
                        model.pt 
            model: Model to load state into.
            strict: Whether to strictly enforce state dict key matching.

        Returns:
            Dictionary with ``missing_keys`` and ``unexpected_keys`` sets.

        Raises:
            CheckpointError: If model file is invalid or corrupted.

        Example:
            >>> info = manager.load_model_only("checkpoints/best/model.pt", model)
            >>> print(f"Missing keys: {info['missing_keys']}")
        """
        path = Path(model_path)

        # Handle both a direct model.pt path and a checkpoint directory.
        if path.is_dir():
            model_file = path / "model.pt"
        else:
            model_file = path

        if not model_file.exists():
            raise CheckpointError(f"Model file not found: {model_file}")

        try:
            model_state = torch.load(model_file, map_location="cpu", weights_only=True)
        except Exception as e:
            raise CheckpointError(f"Failed to load model file {model_file}: {e}") from e

        # Get the actual model (unwrap if using accelerator)
        if self.accelerator is not None:
            target_model = self.accelerator.unwrap_model(model)
        else:
            target_model = model

        # Load with tracking of missing/unexpected keys
        if strict:
            target_model.load_state_dict(model_state, strict=True)
            result: dict[str, set[str]] = {"missing_keys": set(), "unexpected_keys": set()}
        else:
            # Use _load_from_state_dict-style key accounting for non-strict loading.
            current_state = target_model.state_dict()
            missing_keys = set(current_state.keys()) - set(model_state.keys())
            unexpected_keys = set(model_state.keys()) - set(current_state.keys())

            # Load matching keys only
            filtered_state = {k: v for k, v in model_state.items() if k in current_state}
            target_model.load_state_dict(filtered_state, strict=False)

            result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        logger.info(f"Loaded model weights from {model_file}")
        return result

    def validate_checkpoint(self, checkpoint_dir: Path) -> None:
        """Validate checkpoint directory integrity.


        Args:
            checkpoint_dir: Path to checkpoint directory.

        Raises:
            CheckpointError: If checkpoint is invalid or corrupted.
             CheckpointError
        """
        if not checkpoint_dir.exists():
            raise CheckpointError(f"Checkpoint directory not found: {checkpoint_dir}")

        if not checkpoint_dir.is_dir():
            raise CheckpointError(f"Checkpoint path is not a directory: {checkpoint_dir}")

        required_files = ["model.pt", "optimizer.pt", "trainer_state.json"]
        for filename in required_files:
            filepath = checkpoint_dir / filename
            if not filepath.exists():
                raise CheckpointError(
                    f"Missing required file '{filename}' in checkpoint: {checkpoint_dir}"
                )
            if filepath.stat().st_size == 0:
                raise CheckpointError(f"Empty file '{filename}' in checkpoint: {checkpoint_dir}")

        # Try loading files to verify integrity
        try:
            torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
        except Exception as e:
            raise CheckpointError(f"Corrupted model.pt in {checkpoint_dir}: {e}") from e

        try:
            torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        except Exception as e:
            raise CheckpointError(f"Corrupted optimizer.pt in {checkpoint_dir}: {e}") from e

        try:
            with open(checkpoint_dir / "trainer_state.json", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            raise CheckpointError(f"Corrupted trainer_state.json in {checkpoint_dir}: {e}") from e


class CheckpointCallback(Callback):
    """Callback for automatic checkpoint saving during training.


    Args:
        checkpoint_manager: CheckpointManager instance.
        save_steps: Save checkpoint every N steps.
        save_on_epoch_end: Whether to save at epoch end.

    Example:
        >>> manager = CheckpointManager("checkpoints", keep_checkpoints=3)
        >>> callback = CheckpointCallback(manager, save_steps=1000)
        >>> trainer = BaseTrainer(model, callbacks=[callback])
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        save_steps: int | None = None,
        save_on_epoch_end: bool = True,
    ):
        """Initialize checkpoint callback.


        Args:
            checkpoint_manager: CheckpointManager instance.
            save_steps: Save every N steps (None to disable).
            save_on_epoch_end: Save at epoch end.
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_steps = save_steps
        self.save_on_epoch_end = save_on_epoch_end

    def on_step_end(
        self,
        trainer: BaseTrainer,
        step: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Save checkpoint at configured step intervals.

        """
        if self.save_steps is None:
            return

        if step > 0 and step % self.save_steps == 0:
            trainer_state = {
                "metrics": {"train_loss": loss},
                "config": {
                    "max_epochs": trainer.max_epochs,
                    "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
                },
            }
            self.checkpoint_manager.save(
                step=step,
                model=cast(nn.Module, trainer.model),
                optimizer=cast(Optimizer, trainer.optimizer),
                trainer_state=trainer_state,
                epoch=trainer.current_epoch,
            )

    def on_epoch_end(
        self,
        trainer: BaseTrainer,
        epoch: int,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Save checkpoint at epoch end if configured.

         epoch 
        """
        if not self.save_on_epoch_end:
            return

        trainer_state = {
            "metrics": metrics,
            "config": {
                "max_epochs": trainer.max_epochs,
                "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
            },
        }
        self.checkpoint_manager.save(
            step=trainer.global_step,
            model=cast(nn.Module, trainer.model),
            optimizer=cast(Optimizer, trainer.optimizer),
            trainer_state=trainer_state,
            epoch=epoch,
        )
