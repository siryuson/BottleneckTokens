"""Evaluation collator for multimodal embedding models.


This module provides EvalCollator for processing batches of evaluation data.
It delegates model-specific processing to ProcessorWrapper and returns batched inputs
only. Distributed shard/gather ordering stays in evaluator-side protocol helpers.

 EvalCollator
 ProcessorWrapper
 gather  evaluator 

Design
    - Collator receives List[Dict] from LanceDataset semantic runtime samples
    - Images are already decoded by Dataset layer (EvalLanceDataset)
    - New retrieval-schema batches are forwarded to ProcessorWrapper directly
    - Uses batch_processor_outputs to batch samples
    - Returns {"inputs": batched_dict}

Example:
    >>> from vlm2emb.data.collators import EvalCollator
    >>> from vlm2emb.data.processors import create_processor_wrapper
    >>>
    >>> processor_wrapper = create_processor_wrapper(processor=processor)
    >>> collator = EvalCollator(processor_wrapper=processor_wrapper)
    >>> batch = collator(samples)  # samples from DataLoader
    >>> embeddings = model(**batch["inputs"])
"""

from __future__ import annotations

import logging
from typing import Any

from vlm2emb.auto import AutoCollator
from vlm2emb.data.processors import batch_processor_outputs

logger = logging.getLogger(__name__)

@AutoCollator.register("eval")
class EvalCollator:
    """Collator for evaluation datasets.


    Processes batches from LanceDataset for evaluation. Images are expected
    to be already decoded (PIL.Image) by the Dataset layer.

     LanceDataset  Dataset  PIL.Image

    Args:
        processor_wrapper: ProcessorWrapper instance (e.g., Qwen2VLProcessorWrapper)

    Example:
        >>> processor_wrapper = create_processor_wrapper(processor=processor)
        >>> collator = EvalCollator(processor_wrapper=processor_wrapper)
        >>> dataloader = DataLoader(dataset, collate_fn=collator)
        >>> for batch in dataloader:
        ...     embeddings = model(**batch["inputs"])
    """

    def __init__(self, processor_wrapper: Any) -> None:
        self.processor_wrapper = processor_wrapper

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process a batch of evaluation samples.


        Args:
            batch: List of samples from LanceDataset. Each sample should have:
                   - text: str
                   - images: list[PIL.Image] or empty list (already decoded)

        Returns:
            Dict with:
            - "inputs": Batched model inputs from batch_processor_outputs
        """
        if not hasattr(self.processor_wrapper, "process_retrieval_batch"):
            raise TypeError(
                f"{type(self.processor_wrapper).__name__} does not support retrieval schema batches"
            )
        samples = self.processor_wrapper.process_retrieval_batch(batch)

        # Batch samples
        inputs = batch_processor_outputs(samples, self.processor_wrapper)

        return {"inputs": inputs}

    def __repr__(self) -> str:
        return f"EvalCollator(processor_wrapper={type(self.processor_wrapper).__name__})"


__all__ = ["EvalCollator"]
