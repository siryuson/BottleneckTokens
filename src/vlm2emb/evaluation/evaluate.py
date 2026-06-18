"""Evaluation entry point for VLM2Emb.

VLM2Emb evaluation entry point.
"""

from pathlib import Path
from typing import Any, cast

import torch
from accelerate import Accelerator

from vlm2emb.auto import AutoBenchmark
from vlm2emb.config import to_native_config
from vlm2emb.data.processors import create_processor_wrapper, extract_wrapper_kwargs
from vlm2emb.evaluation.io import save_resolved_config, save_results_payload, timestamp_slug
from vlm2emb.model import (
    VLM2Emb,
    _resolve_backbone_source_path,
    create_model_and_processor,
)
from vlm2emb.utils.logging import RankLogger

logger = RankLogger(__name__)


def _resolve_output_dir(
    checkpoint: str | None,
    output_dir: str | None,
) -> Path:
    """Resolve standalone eval output directory."""
    if output_dir:
        return Path(output_dir).expanduser()

    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.exists():
            return checkpoint_path / "eval" / timestamp_slug()

    return Path("./eval") / timestamp_slug()


def _load_processor(path: str, fallback: str | None = None) -> Any:
    """Load processor from path, with optional fallback path.

    Try AutoProcessor first, fallback to AutoTokenizer.

    Args:
        path: Primary path to load processor from.
        fallback: Optional fallback path if primary fails.

    Returns:
        Loaded processor or tokenizer instance.

    Raises:
        ValueError: If processor cannot be loaded from any path.
    """
    from transformers import AutoProcessor, AutoTokenizer

    candidates = [path] + ([fallback] if fallback else [])
    for p in candidates:
        try:
            return AutoProcessor.from_pretrained(p, trust_remote_code=True)
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(p, trust_remote_code=True)
            except Exception:
                continue
    raise ValueError(
        f"Cannot load processor from {path}"
        + (f" or {fallback}" if fallback else "")
    )


def _load_processor_from_config(config: dict[str, Any]) -> Any:
    """Resolve processor from config without rebuilding the whole model."""
    if "processor" in config and config["processor"] is not None:
        from vlm2emb.auto import AutoProcessorWrapper

        wrapper = AutoProcessorWrapper.from_config(config["processor"])
        return wrapper.processor

    model_config = config.get("model")
    if isinstance(model_config, dict):
        modules = model_config.get("modules")
        if isinstance(modules, dict):
            for mod_cfg in modules.values():
                if isinstance(mod_cfg, dict) and "model_name_or_path" in mod_cfg:
                    return _load_processor(
                        _resolve_backbone_source_path(mod_cfg["model_name_or_path"])
                    )

    raise ValueError(
        "Cannot determine processor from config. Provide config.processor or a "
        "model backbone with model_name_or_path."
    )


def _try_load_peft_config(checkpoint_path: str) -> Any | None:
    """Try reading PEFT config from a local path or Hub repo."""
    try:
        from peft import PeftConfig
    except ImportError:
        return None

    try:
        return PeftConfig.from_pretrained(checkpoint_path)
    except Exception:
        return None


def _load_full_checkpoint(
    checkpoint_path: str,
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load a full VLM2Emb checkpoint and resolve processor."""
    model = VLM2Emb.from_pretrained(checkpoint_path)
    try:
        processor = _load_processor(checkpoint_path)
    except ValueError:
        logger.warning(
            "Processor could not be resolved from checkpoint '%s'; "
            "falling back to config initialization.",
            checkpoint_path,
        )
        processor = _load_processor_from_config(config)
        model.setup(processor=processor)

    logger.info("  Full checkpoint loaded: %s", checkpoint_path)
    return model, processor


def _load_peft_processor(
    checkpoint_path: str,
    base_model_path: str | None,
) -> Any:
    """Load processor for PEFT eval, preferring base_model artifacts."""
    if base_model_path:
        return _load_processor(base_model_path, fallback=checkpoint_path)
    return _load_processor(checkpoint_path)


def _load_peft_checkpoint(
    checkpoint_path: str,
    config: dict[str, Any],
    peft_config: Any,
) -> tuple[Any, Any]:
    """Load a PEFT checkpoint, falling back to config initialization when needed."""
    base_model_path = getattr(peft_config, "base_model_name_or_path", "") or ""

    try:
        from peft import AutoPeftModel

        model = AutoPeftModel.from_pretrained(checkpoint_path)
        model = model.merge_and_unload()
        processor = _load_peft_processor(checkpoint_path, base_model_path or None)
        logger.info("  PEFT checkpoint loaded directly: %s", checkpoint_path)
        return model, processor
    except Exception as exc:
        if "model" not in config or config["model"] is None:
            raise ValueError(
                f"Failed to load PEFT checkpoint '{checkpoint_path}' directly and "
                "config has no model fallback. Provide model/processor config for "
                "fallback initialization."
            ) from exc

        logger.warning(
            "Direct PEFT checkpoint load failed for '%s'; falling back to config "
            "initialization. Error: %s",
            checkpoint_path,
            exc,
        )

    model, processor = create_model_and_processor(config)

    from peft import PeftModel

    peft_model = cast(Any, PeftModel).from_pretrained(model, checkpoint_path)
    model = peft_model.merge_and_unload()

    try:
        processor = _load_peft_processor(checkpoint_path, base_model_path or None)
    except ValueError:
        if processor is None:
            raise
        logger.warning(
            "Processor could not be resolved from checkpoint '%s'; using processor "
            "initialized from config.",
            checkpoint_path,
        )

    logger.info("  PEFT checkpoint loaded via config fallback: %s", checkpoint_path)
    return model, processor


def load_model_and_processor(
    config: Any,
    checkpoint: str | None = None,
) -> tuple[Any, Any, Accelerator]:
    """Resolve runtime model and processor for standalone evaluation."""
    config = cast(dict[str, Any], to_native_config(config, resolve=True))

    if "eval" not in config:
        raise ValueError("Config must contain 'eval' with benchmark configuration.")

    checkpoint_path = checkpoint
    has_model = "model" in config and config["model"] is not None

    if not checkpoint_path and not has_model:
        raise ValueError(
            "Config must contain 'eval' and either a runtime checkpoint argument or "
            "a 'model' section for fallback initialization."
        )

    accelerator = Accelerator()

    if checkpoint_path:
        logger.info("[1/3] Loading checkpoint: %s", checkpoint_path)
        peft_config = _try_load_peft_config(checkpoint_path)
        if peft_config is not None:
            model, processor = _load_peft_checkpoint(checkpoint_path, config, peft_config)
        else:
            model, processor = _load_full_checkpoint(checkpoint_path, config)
    else:
        logger.warning(
            "No runtime checkpoint provided; falling back to model/processor "
            "initialization from config."
        )
        model, processor = create_model_and_processor(config)
        logger.info("  Model built: %s", model.__class__.__name__)

    if processor is None:
        raise ValueError(
            "Cannot determine processor. Ensure checkpoint contains processor files "
            "or provide config.processor / model fallback information."
        )

    model = model.to(accelerator.device)
    model.eval()
    return model, processor, accelerator


def _validate_accelerator(accelerator: Any) -> Accelerator:
    """Validate evaluate() received a full Accelerator-compatible runtime."""
    required_attrs = (
        "device",
        "is_main_process",
        "num_processes",
        "process_index",
        "wait_for_everyone",
        "end_training",
        "autocast",
    )
    missing = [name for name in required_attrs if not hasattr(accelerator, name)]
    if missing:
        raise TypeError(
            "evaluate() requires an accelerate.Accelerator-compatible object; "
            f"missing attributes: {', '.join(missing)}"
        )
    return cast(Accelerator, accelerator)


def evaluate(
    config,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run evaluation from configuration and optional runtime checkpoint."""
    config = cast(dict[str, Any], to_native_config(config, resolve=True))
    accelerator = None

    try:
        model, processor, accelerator = load_model_and_processor(
            config,
            checkpoint=checkpoint,
        )
        accelerator = _validate_accelerator(accelerator)
        logger.info("Evaluating on %s processes", accelerator.num_processes)

        logger.info("[2/3] Loading processor...")
        processor_wrapper = create_processor_wrapper(
            processor=processor,
            **extract_wrapper_kwargs(cast(dict[str, Any] | None, config.get("processor"))),
        )
        logger.info("  Processor: %s", type(processor).__name__)

        logger.info("[3/3] Running benchmark...")
        eval_config = config["eval"]
        benchmark = AutoBenchmark.from_config(eval_config)
        logger.info("  Benchmark: %s (%s evaluators)", benchmark.name, len(benchmark))

        # Retrieval evaluators already scope autocast to model forward during
        # query/candidate encoding. Keep prediction and metric computation out of
        # the outer autocast region so ranking math runs in full precision.
        with torch.no_grad():
            results = benchmark(model, processor_wrapper, accelerator)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            save_dir = _resolve_output_dir(checkpoint, output_dir)
            results_path = save_dir / "results.json"
            resolved_config_path = save_dir / "config.yaml"
            save_results_payload(
                results_path,
                results,
                meta={
                    "checkpoint": checkpoint,
                    "config_path": config_path,
                    "world_size": accelerator.num_processes,
                },
            )
            save_resolved_config(resolved_config_path, config)
            logger.info("Eval results saved to %s", results_path)
            logger.info("Resolved config saved to %s", resolved_config_path)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and "_summary" in results:
            summary = results["_summary"]
            logger.info("Evaluation summary:")
            for key, val in summary.items():
                if isinstance(val, (int, float)):
                    logger.info(f"  {key}: {val:.4f}")
                else:
                    logger.info(f"  {key}: {val}")

        return results
    finally:
        if accelerator is not None:
            try:
                accelerator.wait_for_everyone()
            except Exception as exc:
                logger.warning("Evaluation cleanup barrier failed: %s", exc)
            try:
                accelerator.end_training()
            except Exception as exc:
                logger.warning("Evaluation runtime teardown failed: %s", exc)


__all__ = ["evaluate"]
