"""Generic embedding trainer based on Hugging Face Trainer.

This module provides a generic trainer for embedding models that extends
Hugging Face Trainer with support for (query, target) pair training.
Specific trainers should inherit from this class.

Reference:
    VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks
    https://arxiv.org/abs/2410.05160
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedModel, TrainingArguments
from transformers import Trainer as HF_Trainer
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME, ExportableState
from transformers.trainer_utils import SaveStrategy

from vlm2emb.auto import AutoTrainer, AutoTrainingArgs
from vlm2emb.evaluation.benchmarks.base import BaseBenchmark
from vlm2emb.evaluation.io import save_results_payload
from vlm2emb.training.samplers import BatchInterleaveSampler

logger = logging.getLogger(__name__)


# ============================================================================
# Training arguments.
# ============================================================================

# Register Hugging Face TrainingArguments as the default training-args class.
AutoTrainingArgs.register("default", TrainingArguments)


# ============================================================================
# Trainer.
# ============================================================================


@AutoTrainer.register("trainer")
class Trainer(HF_Trainer):
    """Generic trainer for embedding models.

    This trainer handles contrastive learning for embedding models with
    (qry, pos) pair format. It extends HuggingFace Trainer with:
    - Support for (qry, pos) tuple inputs
    - Custom dataloader for IterableDataset
    - Flexible checkpoint saving/loading

    Attributes:
        is_distributed: Whether running in distributed mode
        processor: The multimodal processor
        loss_fn: Contrastive loss function

    Example:
        >>> from transformers import TrainingArguments
        >>> from vlm2emb.training.trainer import EmbeddingTrainer
        >>>
        >>> args = TrainingArguments(
        ...     output_dir="./output",
        ...     per_device_train_batch_size=8,
        ... )
        >>> trainer = EmbeddingTrainer(
        ...     model=model,
        ...     args=args,
        ...     train_dataset=dataset,
        ...     data_collator=collator,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: Any | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list | None = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Callable | None = None,
        **kwargs,
    ):
        """Initialize the embedding trainer.

        Args:
            model: The model to train
            args: Training arguments (TrainingArguments or subclass with temperature)
            data_collator: Data collator for batching
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Processor/tokenizer for the model
            model_init: Function to initialize model
            compute_metrics: Function to compute metrics
            callbacks: Training callbacks
            optimizers: Tuple of (optimizer, scheduler)
            preprocess_logits_for_metrics: Function to preprocess logits
            **kwargs: Additional arguments (for future extensibility)
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Distributed training info is available via self.accelerator when needed.
        # self.is_distributed = self.accelerator.use_distributed
        # self._dist_loss_scale_factor = self.accelerator.num_processes

        # Set up the default contrastive loss.
        self._setup_loss_fn()

    def _log_batch_dataset_sources(self, inputs: dict[str, Any]) -> None:
        """Log dataset sources for current batch by rank order.

        Uses accelerator for distributed coordination.

        Args:
            inputs: Dict with "query", "positive", "negative" and extra fields
        """
        # Collator passes extra fields as top-level keys, e.g. inputs["dataset_name"].
        dataset_names = inputs.get("dataset_name", [])
        if not dataset_names:
            metadata = inputs.get("metadata", {})
            if isinstance(metadata, dict):
                dataset_names = metadata.get("dataset_name", [])

        if not dataset_names:
            return

        # Count dataset sources in this batch.
        dataset_counts = Counter(name if name else "unknown" for name in dataset_names)

        # Get rank info from accelerator.
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        # Format dataset sources as "dataset1:count, dataset2:count, ...".
        sources_str = ", ".join(
            f"{name}:{count}" for name, count in sorted(dataset_counts.items())
        )

        # Collect sequence lengths and visual-field diagnostics per group.
        len_parts = []
        anomaly_parts = []

        # Get vision token IDs from processing_class if available.
        video_token_id = getattr(self.processing_class, "video_token_id", None)
        image_token_id = getattr(self.processing_class, "image_token_id", None)

        for group in ("query", "positive", "negative"):
            samples = inputs.get(group, [])
            if not samples:
                continue
            seq_lens = [
                s["input_ids"].shape[-1] for s in samples
                if isinstance(s, dict) and "input_ids" in s
            ]
            if seq_lens:
                len_parts.append(
                    f"{group}: n={len(seq_lens)} len=[{min(seq_lens)}, {max(seq_lens)}] avg={sum(seq_lens)/len(seq_lens):.0f}"
                )

            # Check for mismatches between visual tokens and visual tensors.
            n_video_tok_no_pv = 0  # has <|video_pad|> in input_ids but pixel_values_videos is None
            n_image_tok_no_pv = 0  # has <|image_pad|> in input_ids but pixel_values is None
            n_pv_video_no_grid = 0  # has pixel_values_videos but no video_grid_thw
            n_grid_no_pv_video = 0  # has video_grid_thw but no pixel_values_videos

            for s in samples:
                if not isinstance(s, dict):
                    continue
                ids = s.get("input_ids")
                has_video_tok = video_token_id is not None and ids is not None and (ids == video_token_id).any().item()
                has_image_tok = image_token_id is not None and ids is not None and (ids == image_token_id).any().item()
                has_pv_video = s.get("pixel_values_videos") is not None
                has_pv_image = s.get("pixel_values") is not None
                has_video_grid = s.get("video_grid_thw") is not None

                if has_video_tok and not has_pv_video:
                    n_video_tok_no_pv += 1
                if has_image_tok and not has_pv_image:
                    n_image_tok_no_pv += 1
                if has_pv_video and not has_video_grid:
                    n_pv_video_no_grid += 1
                if has_video_grid and not has_pv_video:
                    n_grid_no_pv_video += 1

            if n_video_tok_no_pv:
                anomaly_parts.append(f"{group}: {n_video_tok_no_pv} video_tok_no_pixels")
            if n_image_tok_no_pv:
                anomaly_parts.append(f"{group}: {n_image_tok_no_pv} image_tok_no_pixels")
            if n_pv_video_no_grid:
                anomaly_parts.append(f"{group}: {n_pv_video_no_grid} video_pixels_no_grid")
            if n_grid_no_pv_video:
                anomaly_parts.append(f"{group}: {n_grid_no_pv_video} grid_no_video_pixels")

        lens_str = " | ".join(len_parts)

        logger.info("[%d/%d] Batch sources: %s | %s", rank, world_size, sources_str, lens_str)
        if anomaly_parts:
            logger.warning(
                "[%d/%d] ⚠ Visual anomalies: %s",
                rank,
                world_size,
                " | ".join(anomaly_parts),
            )

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> Sampler | None:
        """Get training sampler with BatchInterleaveSampler support.

        If dataset has metadata (dataset_lengths, dataset_offsets, weights),
        uses BatchInterleaveSampler for batch-wise interleaving.
        Otherwise, falls back to default Hugging Face behavior.

        Args:
            train_dataset: Training dataset.

        Returns:
            Sampler instance or None.
        """
        if train_dataset is None:
            train_dataset = cast(Dataset | None, self.train_dataset)
        if train_dataset is None:
            return None

        # Check if dataset has metadata for batch interleaving.
        interleave_batch_size = getattr(self.args, "interleave_batch_size", 0)

        if interleave_batch_size > 0 and hasattr(train_dataset, "dataset_lengths"):
            stopping_strategy = cast(
                Literal["all_exhausted", "first_exhausted"],
                getattr(self.args, "interleave_stopping_strategy", "all_exhausted"),
            )
            shuffle_within_dataset = getattr(
                self.args, "interleave_shuffle_within_dataset", True
            )
            logger.info(
                f"Using BatchInterleaveSampler with batch_size={interleave_batch_size}, "
                f"stopping_strategy={stopping_strategy}, "
                f"shuffle_within_dataset={shuffle_within_dataset}"
            )
            return BatchInterleaveSampler(
                dataset=train_dataset,
                batch_size=interleave_batch_size,
                stopping_strategy=stopping_strategy,
                shuffle_within_dataset=shuffle_within_dataset,
                seed=self.args.seed,
                per_device_batch_size=self.args.per_device_train_batch_size,
                num_processes=self.args.world_size,
            )

        # Fall back to default behavior (RandomSampler).
        return super()._get_train_sampler(train_dataset)

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Override to set multiprocessing_context='spawn' for CUDA safety.

        When num_workers > 0, PyTorch DataLoader workers must use 'spawn'
        start method to avoid CUDA re-initialization issues in forked processes.
        The accelerate DataLoaderShard delegates __getattr__ to base_dataloader
        but __setattr__ writes to its own __dict__, so we must set on
        base_dataloader directly.
        """
        dataloader = super().get_train_dataloader()

        if self.args.dataloader_num_workers > 0:
            base = getattr(dataloader, "base_dataloader", dataloader)
            base.multiprocessing_context = "spawn"

        return dataloader

    def _setup_loss_fn(self) -> None:
        """Setup contrastive loss function.

        Override this method in subclasses for custom loss functions.
        """
        from vlm2emb.training.losses.contrastive import (
            ContrastiveLoss,
            DistributedContrastiveLoss,
        )

        if self.accelerator.use_distributed:
            temperature = float(getattr(self.args, "temperature", 0.02))
            self.loss_fn = DistributedContrastiveLoss(temperature=temperature)
        else:
            temperature = float(getattr(self.args, "temperature", 0.02))
            self.loss_fn = ContrastiveLoss(temperature=temperature)

    # ========================================================================
    # Core training methods.
    # ========================================================================

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, dict],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute contrastive loss for (qry, pos) pairs.

        Args:
            model: The model
            inputs: Dict with "query", "positive", "negative" (optional), "metadata" keys
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)

        Returns:
            Loss tensor, optionally with outputs
        """
        # Extract query and positive inputs from the collated batch.
        qry_inputs = inputs["query"]
        pos_inputs = inputs["positive"]
        # neg_inputs = inputs.get("negative")  # For future hard negative mining

        # Remove metadata before forward pass; the model does not consume it.
        qry_inputs = {k: v for k, v in qry_inputs.items() if k != "metadata"}
        pos_inputs = {k: v for k, v in pos_inputs.items() if k != "metadata"}

        # Forward pass for queries and positives; use forward, not encode.
        qry_outputs = model(**qry_inputs)
        pos_outputs = model(**pos_inputs)

        qry_embeds = qry_outputs["embeddings"]
        pos_embeds = pos_outputs["embeddings"]

        # Compute contrastive loss.
        loss = self.loss_fn(qry_embeds, pos_embeds)

        if return_outputs:
            return loss, {"qry_embeds": qry_embeds, "pos_embeds": pos_embeds}
        return loss

    # ========================================================================
    # Evaluation override.
    # ========================================================================

    def evaluate(
        self,
        eval_dataset: Dataset | BaseBenchmark | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate model, routing to BaseBenchmark if applicable.

        This method overrides HF Trainer.evaluate() to support custom benchmarks.
        If eval_dataset is a BaseBenchmark subclass, delegates to benchmark.__call__().
        Otherwise raises NotImplementedError (future: fallback to parent).

        Args:
            eval_dataset: Evaluation dataset or benchmark
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric names (default: "eval")

        Returns:
            Dictionary of metric scores
        """
        from vlm2emb.evaluation.benchmarks.base import BaseBenchmark

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if dataset is None:
            logger.warning("No eval_dataset provided, skipping evaluation")
            return {}

        # Route to BaseBenchmark if applicable.
        if isinstance(dataset, BaseBenchmark):
            logger.info(f"Evaluating with {dataset.__class__.__name__}")

            self._memory_tracker.start()

            eval_model = cast(nn.Module | None, self.model)
            if eval_model is None:
                raise ValueError("Trainer.model is not initialized")
            eval_model.eval()

            start_time = time.time()
            results = dataset(
                model=eval_model,
                processor_wrapper=self.processing_class,
                accelerator=self.accelerator,
            )
            elapsed = time.time() - start_time

            runtime = round(self._distributed_max_float(elapsed), 2)
            metrics = self._flatten_benchmark_results(results, metric_key_prefix)
            metrics[f"{metric_key_prefix}_runtime"] = runtime

            # Match HF Trainer.evaluate(): every rank logs and runs callbacks so
            # TrainerControl flags are reset consistently after evaluation.
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            self._memory_tracker.stop_and_update_metrics(metrics)

            # Only the main process writes evaluation artifacts.
            if self.accelerator.is_main_process:
                self._save_eval_results(results)

            return metrics

        # Other evaluation dataset types are not supported by this public trainer yet.
        raise NotImplementedError(
            f"Unsupported eval_dataset type: {type(dataset).__name__}. "
            f"Expected BaseBenchmark subclass."
        )

    def _distributed_max_float(self, value: float) -> float:
        """Return the max scalar value across ranks through Accelerate gather."""
        num_processes = getattr(self.accelerator, "num_processes", 1)
        if num_processes <= 1:
            return value

        gather = getattr(self.accelerator, "gather", None)
        if gather is None:
            return value

        device = getattr(self.accelerator, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([value], device=device, dtype=torch.float64)
        gathered = gather(tensor)
        return float(gathered.max().item())

    def _flatten_benchmark_results(
        self,
        results: Any,
        prefix: str,
    ) -> dict[str, float]:
        """Flatten benchmark results to metric dict.

        Reads from results["_summary"] (BaseBenchmark convention).

        Args:
            results: Benchmark results dict with "_summary" key
            prefix: Metric key prefix

        Returns:
            Flattened metric dict {prefix_key: value}
        """
        # Convert to dict if needed
        if hasattr(results, "to_dict"):
            results = results.to_dict()

        flat: dict[str, float] = {}

        # Read from "_summary" (BaseBenchmark convention)
        summary = results.get("_summary", {})
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                flat[f"{prefix}_{key}"] = value

        return flat

    def _save_eval_results(
        self,
        results: dict[str, Any],
    ) -> None:
        """Save evaluation results to an eval-only path.

        Saves to {output_dir}/eval-results/step-{global_step}/eval_results.json.
        This deliberately avoids creating checkpoint-{global_step} during
        evaluation-only steps, because HuggingFace Trainer treats checkpoint
        directory existence as a signal when resolving best_model_checkpoint.

        Args:
            results: Full benchmark results (per-dataset + _summary)
        """
        if self.args.output_dir is None:
            raise ValueError("TrainingArguments.output_dir must be set for eval result saving")
        output_dir = Path(self.args.output_dir)
        global_step = self.state.global_step
        eval_dir = output_dir / "eval-results" / f"step-{global_step}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        save_path = eval_dir / "eval_results.json"
        save_results_payload(
            save_path,
            results,
            meta={"global_step": global_step},
        )

        logger.info(f"Eval results saved to {save_path}")

    def _save_checkpoint(self, model: nn.Module, trial: Any) -> None:
        """Save checkpoint with accelerate-managed synchronization.

        This overrides the upstream HF implementation to avoid calling a raw
        ``torch.distributed.barrier()`` without ``device_ids`` during
        checkpoint saves. The logic is otherwise kept aligned with
        transformers 4.57.3.

        Keep this method in sync with upstream ``transformers.Trainer`` when
        upgrading transformers, and preserve the accelerate-managed barrier
        unless upstream fixes the device binding behavior.
        """
        # NOTE: This method intentionally mirrors transformers 4.57.3 with one
        # targeted change: replace the raw distributed barrier in checkpoint
        # save flow so PyTorch does not warn about missing device binding.
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            # Keep checkpoint visibility synchronization under Accelerator so
            # the underlying barrier uses the correct local device binding.
            self.accelerator.wait_for_everyone()

            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            self._save_rng_state(output_dir)

        if self.args.should_save:
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
