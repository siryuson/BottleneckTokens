"""Training orchestration for BToks and baseline retrieval models.

This module provides the orchestration layer that builds components
and creates trainers following the HuggingFace pattern:
- train.py: Builds Model, Dataset, Collator (orchestration)
- Trainer: Creates DataLoader, Optimizer, Scheduler, runs training loop (execution)

    Logging:
    This module uses distributed-aware logging. When running with multiple GPUs,
    logs are only printed from rank 0 (main process) to avoid duplicates.

    To enable WandB tracking, add to your training config:

        training:
          report_to: "wandb"  # or ["wandb", "tensorboard"]
          run_name: "my-experiment"

    To disable tracking for debugging:

        WANDB_MODE=disabled python scripts/train.py config.yaml
"""

import logging
import os
from datetime import timedelta
from typing import Any, cast

import torch.nn as nn
from accelerate import PartialState

from vlm2emb.auto import (
    AutoBenchmark,
    AutoCollator,
    AutoDataset,
    AutoTrainer,
    AutoTrainingArgs,
)
from vlm2emb.config import Config, to_native_config

logger = logging.getLogger(__name__)

DEFAULT_DDP_TIMEOUT_SECONDS = 1800


def _get_ddp_timeout_seconds(training_args: Any) -> int:
    """Return the configured DDP process group timeout in seconds."""
    timeout = getattr(training_args, "ddp_timeout", DEFAULT_DDP_TIMEOUT_SECONDS)
    if timeout is None:
        return DEFAULT_DDP_TIMEOUT_SECONDS
    return int(timeout)


# ============================================================================
# Eval config validation.
# ============================================================================


def _validate_eval_config(eval_config: dict[str, Any]) -> None:
    """Validate eval config before benchmark/dataset construction."""
    if "type" not in eval_config or not eval_config["type"]:
        raise ValueError(
            "Eval config exists but is missing required 'type'. "
            "If you only want to override eval settings, start from a preset with "
            "an eval section or also provide eval.type=... ."
        )


# ============================================================================
# PEFT / LoRA Management
# ============================================================================

def apply_peft(model: nn.Module, peft_config: dict) -> nn.Module:
    """Apply PEFT (LoRA, etc.) to model.

    This is the central place for PEFT application, keeping the retrieval model
    assembly independent from adapter setup.

    Args:
        model: The retrieval model.
        peft_config: PEFT configuration dict with keys:
            - type: "lora", "qlora", "ia3", etc.
            - r: LoRA rank.
            - alpha: LoRA alpha.
            - target_modules: Target modules.
            - dropout: Dropout rate.
            - use_dora: Whether to use DoRA.

    Returns:
        PEFT-wrapped model.

    Example config:
        peft:
          type: lora
          r: 16
          alpha: 64
          target_modules: "q_proj,k_proj,v_proj,o_proj"
          dropout: 0.1
          use_dora: true
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as err:
        raise ImportError(
            "PEFT is required for LoRA/PEFT training. Install with: pip install peft"
        ) from err

    peft_type = peft_config.get("type", "lora")

    if peft_type == "lora":
        # Parse target modules.
        target_modules = peft_config.get(
            "target_modules", "qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj"
        ).split(",")

        # Infer modules_to_save unless user specified it.
        modules_to_save = peft_config.get("modules_to_save")
        if modules_to_save is None and peft_config.get("auto_infer_modules_to_save", True):
            modules_to_save = cast(Any, model).infer_modules_to_save() or None
        if modules_to_save:
            logger.info("  modules_to_save: %s", modules_to_save)

        lora_config = LoraConfig(
            r=peft_config.get("r", 16),
            lora_alpha=peft_config.get("alpha", 64),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            lora_dropout=peft_config.get("dropout", 0.1),
            init_lora_weights=peft_config.get("init_weights", "gaussian"),
            use_dora=peft_config.get("use_dora", True),
            inference_mode=False,
        )

        # Apply trainable_token_indices from model protocol.
        trainable_token_indices = peft_config.get("trainable_token_indices")
        if trainable_token_indices is not None:
            lora_config.trainable_token_indices = trainable_token_indices
            logger.info("  trainable_token_indices: %s", trainable_token_indices)

        logger.info(
            "Applying LoRA: r=%s, alpha=%s, target_modules=%s, use_dora=%s",
            lora_config.r,
            lora_config.lora_alpha,
            target_modules,
            lora_config.use_dora,
        )

        model = cast(nn.Module, get_peft_model(cast(Any, model), lora_config))
        cast(Any, model).print_trainable_parameters()

    return model


# ============================================================================
# Training output
# ============================================================================


def train(config: Config | dict[str, Any]) -> None:
    """Main training entry point.

    Orchestrates training by:
    1. Building Model from config
    2. Building Dataset(s) from config
    3. Building DataCollator from config
    4. Creating Trainer with all components
    5. Calling trainer.train()

    Args:
        config: Full training configuration.

    Example:
        >>> from vlm2emb.config import load_config
        >>> config = load_config("configs/experiments/vlm2vec.yaml")
        >>> train(config)
    """
    config = cast(dict[str, Any], to_native_config(config, resolve=True))

    # Use AutoTrainingArgs to create the appropriate TrainingArguments subclass.
    train_config = config["train"]
    training_args = AutoTrainingArgs.from_config(train_config["args"])
    ddp_timeout_seconds = _get_ddp_timeout_seconds(training_args)

    # Initialize distributed state after TrainingArguments is available so the
    # process-group timeout comes from train.args.ddp_timeout.
    state = PartialState(timeout=timedelta(seconds=ddp_timeout_seconds))
    rank = state.process_index
    world_size = state.num_processes

    logger.info("Starting Training Pipeline")
    logger.debug(
        "Distributed setup: rank=%s, world_size=%s, ddp_timeout=%ss",
        rank,
        world_size,
        ddp_timeout_seconds,
    )
    logger.debug("  TrainingArgs type: %s", type(training_args).__name__)
    logger.debug("  Output dir: %s", training_args.output_dir)

    # ==================== 1. Build Model + Processor ====================
    logger.info("[1/5] Building model and processor...")

    from vlm2emb.model import create_model_and_processor

    model, processing_class = create_model_and_processor(config)
    logger.info("  Model type: %s", config["model"]["type"])
    if processing_class is not None:
        logger.info("  Processor: %s", type(processing_class).__name__)

    if processing_class is not None:
        # Sync special token IDs from processor to model config.
        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            model.config.bos_token_id = tokenizer.bos_token_id

    logger.debug("[%s/%s] Model built and prepared: %s", rank, world_size, model.__class__.__name__)
    state.wait_for_everyone()

    # ==================== 2. Apply PEFT (if configured) ====================
    # PEFT config can be in:
    # - config.peft (top-level)
    # - config.training.peft (under training)
    peft_config = None
    if config.get("peft") is not None:
        peft_config = dict(cast(dict[str, Any], config["peft"]))

    if peft_config is not None:
        logger.info("[2/5] Applying PEFT...")

        # Save lightweight base model before PEFT wrapping (reference mode).
        # This enables AutoPeftModel.from_pretrained to load the base model
        # from adapter_config.json's base_model_name_or_path.
        base_model_path = os.path.join(training_args.output_dir, "base_model")
        if state.is_main_process:
            model.save_pretrained(base_model_path, save_base_weights=False)
            # Save full processor (image_processor + tokenizer) for eval loading.
            if processing_class is not None:
                processing_class.save_pretrained(base_model_path)
                logger.info("  Saved processor to %s", base_model_path)
            logger.info("  Saved base model (reference) to %s", base_model_path)
        state.wait_for_everyone()

        # Set name_or_path so PEFT writes a meaningful base_model_name_or_path
        # in adapter_config.json (fixes empty string warning).
        model.__dict__["name_or_path"] = base_model_path
        model.config._name_or_path = base_model_path

        # Collect trainable token indices from model (generic protocol).
        trainable_token_indices = model.get_trainable_token_indices()
        if trainable_token_indices is not None:
            peft_config["trainable_token_indices"] = trainable_token_indices
            logger.info("  trainable_token_indices (from model): %s", trainable_token_indices)

        model = apply_peft(model, peft_config)
    else:
        logger.info("[2/5] PEFT not configured, using full model")
        # Log trainable parameters for full model.
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  Total parameters: %s", f"{total_params:,}")
        logger.info("  Trainable parameters: %s", f"{trainable_params:,}")

    logger.debug(
        "[%s/%s] Model after PEFT applied: %s, trainable_params=%s",
        rank,
        world_size,
        model.__class__.__name__,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ==================== 3. Build Dataset ====================
    state.wait_for_everyone()
    logger.info("[3/5] Building datasets...")
    # Train dataset (required)
    train_dataset_config = train_config["dataset"]
    collator_config = train_config.get("collator")

    train_dataset = AutoDataset.from_config(train_dataset_config)

    # IterableDataset may not have __len__.
    train_size = len(train_dataset) if hasattr(train_dataset, "__len__") else "unknown"
    logger.info(
        "  Train dataset: %s (%s), size=%s",
        train_dataset_config.get("type", "unknown"),
        type(train_dataset).__name__,
        train_size,
    )

    logger.debug(
        "Train dataset type: %s, has_len: %s, size: %s",
        type(train_dataset).__name__,
        hasattr(train_dataset, "__len__"),
        train_size,
    )

    # Eval dataset is optional; try AutoBenchmark first, then fallback to AutoDataset.
    eval_dataset = None
    if config.get("eval") is not None:
        eval_config = cast(dict[str, Any], config["eval"])
        _validate_eval_config(eval_config)
        eval_type = eval_config.get("type", "")

        # Import benchmarks to trigger registration.
        import vlm2emb.evaluation.benchmarks  # noqa: F401

        if eval_type and eval_type in AutoBenchmark:
            eval_dataset = AutoBenchmark.from_config(eval_config)
            logger.info("  Eval benchmark: %s (%s)", eval_type, type(eval_dataset).__name__)
        else:
            eval_dataset = AutoDataset.from_config(eval_config)
            eval_size = len(eval_dataset) if hasattr(eval_dataset, "__len__") else "unknown"
            logger.info("  Eval dataset: %s, size=%s", eval_type, eval_size)

    logger.debug("[%s/%s] Eval dataset built", rank, world_size)

    # ==================== 4. Build Collator ====================

    state.wait_for_everyone()
    logger.info("[4/5] Building data collator...")
    # Collator config was extracted from train.collator.
    if collator_config is None:
        raise ValueError("Missing 'collator' config under train")

    # Import collators and processors to trigger registration.
    import vlm2emb.data.collators  # noqa: F401
    import vlm2emb.data.processors  # noqa: F401

    # Create ProcessorWrapper from processor for collator and trainer.
    from vlm2emb.data.processors import create_processor_wrapper, extract_wrapper_kwargs

    wrapper = None
    if processing_class is not None:
        wrapper = create_processor_wrapper(
            processor=processing_class,
            **extract_wrapper_kwargs(cast(dict[str, Any] | None, config.get("processor"))),
        )
        logger.info("  ProcessorWrapper: %s", type(wrapper).__name__)

    data_collator = AutoCollator.from_config(collator_config, wrapper=wrapper)
    logger.info("  Collator type: %s", collator_config.get("type", "unknown"))

    # ==================== 5. Build Trainer ====================
    state.wait_for_everyone()
    logger.info("[5/5] Building trainer...")

    # Log tracking configuration.
    report_to = training_args.report_to
    if report_to and report_to != ["none"]:
        logger.info("  Experiment tracking: %s", report_to)
        if "wandb" in (report_to if isinstance(report_to, list) else [report_to]):
            run_name = getattr(training_args, "run_name", None)
            if run_name:
                logger.info("  WandB run name: %s", run_name)
    else:
        logger.info("  Experiment tracking: disabled (set report_to to enable)")

    logger.debug("  Full training arguments: %s", training_args)

    # Trainer API follows HuggingFace Trainer pattern
    # Trainer-specific params (temperature, use_grad_cache, etc.) are now in training_args
    trainer_config = dict(cast(dict[str, Any], train_config["trainer"]))
    # Remove training_args from config since it's already passed as 'args'.
    trainer_config.pop("training_args", None)

    trainer = AutoTrainer.from_config(
        trainer_config,
        # __init__ parameters
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=wrapper or processing_class,
    )
    logger.info("  Trainer type: %s", train_config["trainer"].get("type"))

    logger.debug("[%s/%s] Trainer built", rank, world_size)

    # ==================== 6. Train ====================
    logger.info("Starting training...")

    # Resume from checkpoint if specified.
    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    if resume_from_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)

    # trainer.train() handles everything:
    # - DataLoader creation
    # - Optimizer/Scheduler creation
    # - Distributed setup (DDP/DeepSpeed/FSDP)
    # - Training loop
    # - Checkpointing
    # - Logging
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ==================== 7. Finalize Distributed Training ====================
    trainer.accelerator.wait_for_everyone()
    logger.info("Training completed!")
    trainer.accelerator.end_training()
