"""Processor wrappers for different VLM backends.

This module provides ProcessorWrappers that wrap transformers processors to provide
unified input/output interface for different VLM models.

Design
    - ProcessorWrappers return List[Dict], one dict per sample (not batched)
    - Use batch_processor_outputs() to batch samples before model forward
    - This design enables flexible batching in trainer (e.g., GradCache chunks)

Available wrappers
    - Qwen2VLProcessorWrapper: For Qwen2-VL and Qwen2.5-VL models
    - Qwen3VLProcessorWrapper: For Qwen3-VL models

Auto-creation
    - PROCESSOR_CLASS_MAP: Mapping from processor/tokenizer class name to wrapper registry name
    - create_processor_wrapper(processor, tokenizer, **kwargs): Factory function

Example:
    >>> from vlm2emb.data.processors import create_processor_wrapper
    >>> from transformers import AutoProcessor
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> wrapper = create_processor_wrapper(processor=processor)
    >>> samples = wrapper(texts, images)
"""

from typing import Any

from vlm2emb.auto import AutoProcessorWrapper

from .batch import batch_processor_outputs
from .qwen2_5_vl import Qwen2_5VLProcessorWrapper
from .qwen2_vl import Qwen2VLProcessorWrapper
from .qwen3_vl import Qwen3VLProcessorWrapper

# Mapping from processor/tokenizer class name to wrapper registry name.
PROCESSOR_CLASS_MAP: dict[str, str] = {
    "Qwen2VLProcessor": "qwen2_vl",
    "Qwen2_5_VLProcessor": "qwen2_5_vl",
    "Qwen3VLProcessor": "qwen3_vl",
}

PROCESSOR_RUNTIME_CONFIG_KEYS: set[str] = {
    "type",
    "processor_name",
    "processor",
    "trust_remote_code",
    "min_pixels",
    "max_pixels",
    "image_min_pixels",
    "image_max_pixels",
    "video_min_pixels",
    "video_max_pixels",
    "size",
}
"""Keys consumed while loading the raw transformers processor.

These fields should not be forwarded when wrapping an already constructed
processor instance, otherwise wrapper construction would receive unexpected
kwargs unrelated to runtime behavior toggles.
"""


def extract_wrapper_kwargs(processor_config: dict[str, Any] | None) -> dict[str, Any]:
    """Return wrapper-only kwargs from one processor config block.

    This helper preserves runtime toggles such as
    ``wrap_visual_tokens_with_boundaries`` and ``fps`` while dropping keys that
    were only meant for ``AutoProcessor.from_pretrained(...)``.
    """
    if not processor_config:
        return {}
    return {
        key: value
        for key, value in processor_config.items()
        if key not in PROCESSOR_RUNTIME_CONFIG_KEYS
    }


def create_processor_wrapper(
    processor: Any = None,
    tokenizer: Any = None,
    **kwargs,
) -> Any:
    """Create ProcessorWrapper from processor or tokenizer instance.

    Automatically infers the wrapper type from the processor/tokenizer class name.
    For multimodal models, pass processor; for text-only models, pass tokenizer.

    Args:
        processor: transformers processor instance (e.g., Qwen2VLProcessor).
                   If provided, takes priority for type matching.
        tokenizer: transformers tokenizer instance. Used when no processor available.
        **kwargs: Additional arguments passed to wrapper (e.g., max_length)

    Returns:
        ProcessorWrapper instance

    Raises:
        ValueError: If neither processor nor tokenizer is provided,
                    or if the type is not supported

    Example:
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> wrapper = create_processor_wrapper(processor=processor)
        >>> samples = wrapper(texts, images)
    """
    if processor is None and tokenizer is None:
        raise ValueError("Either 'processor' or 'tokenizer' must be provided")

    # Determine type from processor first, then tokenizer.
    if processor is not None:
        class_name = type(processor).__name__
        wrapper_type = PROCESSOR_CLASS_MAP.get(class_name)
    else:
        class_name = type(tokenizer).__name__
        wrapper_type = PROCESSOR_CLASS_MAP.get(class_name)

    if wrapper_type is None:
        supported = list(PROCESSOR_CLASS_MAP.keys())
        raise ValueError(
            f"Unsupported type: {class_name}. Supported types: {supported}"
        )

    wrapper_cls = AutoProcessorWrapper.get(wrapper_type)

    # Suppress fast tokenizer .pad() advice — our collator uses .pad() by design
    # (input_ids are already encoded, no raw text available for __call__)
    tok = getattr(processor, "tokenizer", processor or tokenizer)
    if hasattr(tok, "deprecation_warnings"):
        tok.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # Pass processor if available, otherwise construct with tokenizer
    if processor is not None:
        return wrapper_cls(processor=processor, **kwargs)
    else:
        return wrapper_cls(tokenizer=tokenizer, **kwargs)


__all__ = [
    "AutoProcessorWrapper",
    "Qwen2VLProcessorWrapper",
    "Qwen2_5VLProcessorWrapper",
    "Qwen3VLProcessorWrapper",
    "batch_processor_outputs",
    "create_processor_wrapper",
    "PROCESSOR_CLASS_MAP",
    "extract_wrapper_kwargs",
]
