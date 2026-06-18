"""Training collator for multimodal embedding models.


This module provides TrainingCollator for processing batches of query-positive-negative
triplets during training. It delegates model-specific processing to ProcessorWrappers.

 TrainingCollator query-positive-negative 
 ProcessorWrapper

Architecture
============================================
    With Lance datasets, media contents are already decoded in Dataset.__getitem__():

    PyTorch Dataset (Lance) yields data with PIL.Image:
        {
            "query": {"text": "...", "media": [{"kind": "image", "content": PIL.Image}]},
            "positive": {"text": "...", "media": [...]},
            "negative": {"text": "...", "media": [...]},  # Empty text = no negative
            "dataset_name": "..."  # optional passthrough metadata
        }
                            ↓
    Collator (simplified, no image loading):
        ProcessorWrapper processes -> List[Dict] (one dict per sample, NOT batched)
                            ↓
    Trainer (handles batching):
        Use batch_processor_outputs() to batch samples before model forward

Design
    - Core encoding sample is {"text", "media", "media_metadata?"}
    - query/positive/negative belong to relation layer, not sample layer
    - dataset_name is passthrough metadata outside the core sample
    - Collator returns List[Dict] per field, not batched tensors
    - This enables flexible batching in trainer (e.g., GradCache chunks)
    - Use batch_processor_outputs() from vlm2emb.data.processors to batch
    - Negative field: None if text is empty, otherwise processed by wrapper
    - Media contents are already decoded by the dataset (no loading needed)

    -  {"text", "media", "media_metadata?"}
    - query
    - dataset_name  sample 
    - Collator  List[Dict]
    -  trainer 
    -  vlm2emb.data.processors  batch_processor_outputs() 
    - Negative  text  None wrapper 

Example:
    >>> from vlm2emb.data.processors import batch_processor_outputs, create_processor_wrapper
    >>>
    >>> wrapper = create_processor_wrapper(processor=processor)
    >>> collator = TrainingCollator(wrapper=wrapper)
    >>> outputs = collator(batch)
    >>> # outputs["query"] is List[Dict], one per sample
    >>> # outputs["negative"] is List[Dict | None], None if no negative
    >>> # Batch in trainer before model forward:
    >>> batched_query = batch_processor_outputs(outputs["query"], wrapper)
"""

from __future__ import annotations

import logging
from typing import Any

from vlm2emb.auto import AutoCollator

logger = logging.getLogger(__name__)


def _parse_bool_config(value: Any, *, field_name: str) -> bool:
    """Parse bool config values from YAML or CLI overrides."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean, got {value!r}")


def _field_instruction_config(
    instruction_policy: dict[str, Any],
    field_name: str,
) -> dict[str, Any]:
    """Return normalized instruction config for one train field."""
    field_cfg = instruction_policy.get(field_name)
    if isinstance(field_cfg, dict):
        return field_cfg

    if field_name == "query":
        instruction = instruction_policy.get("query_instruction")
    else:
        instruction = instruction_policy.get("corpus_instruction")

    cfg: dict[str, Any] = {}
    if instruction:
        cfg["instruction"] = instruction
    return cfg


def render_instruction_text(
    text: str,
    *,
    field_name: str,
    instruction_policy: dict[str, Any] | None,
) -> str:
    """Render GME-style instruction text for a training field.

    The policy is intentionally lightweight and optional. When no instruction
    is configured for a field, the original text is returned unchanged.
    """
    if not instruction_policy or instruction_policy.get("enabled", True) is False:
        return text

    field_cfg = _field_instruction_config(instruction_policy, field_name)
    instruction = str(field_cfg.get("instruction", "") or "").strip()
    if not instruction:
        return text

    template = str(
        field_cfg.get("template")
        or instruction_policy.get("template")
        or "{instruction}\n{text}"
    )
    return template.format(
        instruction=instruction,
        text=text,
        field=field_name,
    )


def _extract_field_media_metadata(
    field_data: dict[str, Any],
    sample: dict[str, Any],
    *,
    field_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return sample-level and legacy relation-level media metadata."""
    field_media_metadata = field_data.get("media_metadata")
    if not isinstance(field_media_metadata, dict):
        field_media_metadata = None

    sample_metadata = sample.get("metadata", {})
    legacy_video_metadata = None
    if isinstance(sample_metadata, dict):
        legacy_video_metadata = sample_metadata.get(f"{field_name}_video_metadata")
    if field_media_metadata is None and not isinstance(legacy_video_metadata, dict):
        legacy_video_metadata = None

    return field_media_metadata, legacy_video_metadata


def _build_train_field_payload(
    field_data: dict[str, Any],
    sample: dict[str, Any],
    *,
    field_name: str,
    text: str,
) -> dict[str, Any]:
    """Build one wrapper-facing training payload without flattening media slots."""
    field_media_metadata, legacy_video_metadata = _extract_field_media_metadata(
        field_data,
        sample,
        field_name=field_name,
    )

    if "media" in field_data:
        media = field_data.get("media") or []
        if field_data.get("images"):
            raise ValueError(
                f"{field_name} contains both MultiModalInput.media and legacy images; "
                "use exactly one media representation"
            )
        if not isinstance(media, list):
            raise TypeError(f"{field_name}.media must be a list")
        payload: dict[str, Any] = {"text": text, "media": media}
        if legacy_video_metadata is not None:
            payload["legacy_media_metadata"] = legacy_video_metadata
    else:
        images = field_data.get("images", [])
        if images is None:
            images = []
        if not isinstance(images, list):
            raise TypeError(f"{field_name}.images must be a list")
        payload = {"text": text, "images": images}
        if legacy_video_metadata is not None:
            payload["media_metadata"] = legacy_video_metadata

    if field_media_metadata is not None:
        payload["media_metadata"] = field_media_metadata

    return payload


@AutoCollator.register("training")
class TrainingCollator:
    """Simplified collator for multimodal training data.


    This collator processes batches of query-positive-negative triplets for contrastive learning.
    Media contents are already decoded from Lance datasets.
    Returns List[Dict] per field for flexible batching in trainer.

     query-positive-negative 
     Lance 
     List[Dict]  trainer 

    Features
        - ProcessorWrapper-based processing for model flexibility
        - Returns List[Dict] per field (NOT batched tensors)
        - Negative field: None if text is empty, otherwise processed
        - No media loading (contents already decoded in Dataset)

    Args:
        wrapper: ProcessorWrapper instance (e.g., Qwen2VLProcessorWrapper).

    Example:
        >>> from vlm2emb.data.processors import batch_processor_outputs, create_processor_wrapper
        >>> wrapper = create_processor_wrapper(processor=processor)
        >>> collator = TrainingCollator(wrapper=wrapper)
        >>> outputs = collator(batch)
        >>> # outputs["query"] is List[Dict], batch in trainer:
        >>> # outputs["negative"] is List[Dict | None], None if no negative
        >>> batched = batch_processor_outputs(outputs["query"], wrapper)
    """

    # Fixed field names for the training schema.
    CORE_FIELDS = ("query", "positive", "negative")

    def __init__(
        self,
        wrapper: Any,
        instruction_policy: dict[str, Any] | None = None,
        use_explicit_negatives: bool = False,
        hard_negative_mining: bool = False,
    ) -> None:
        """Initialize collator.


        Args:
            wrapper: ProcessorWrapper for model-specific processing.
            instruction_policy: Optional GME-style instruction rendering policy.
            use_explicit_negatives: Reserved future flag. Current fair baseline
                keeps explicit negatives disabled.
            hard_negative_mining: Reserved future flag. This collator never
                mines or writes negatives.
        """
        self.wrapper = wrapper
        self.instruction_policy = instruction_policy
        self.use_explicit_negatives = use_explicit_negatives
        self.hard_negative_mining = hard_negative_mining
        if self.use_explicit_negatives:
            raise ValueError(
                "use_explicit_negatives is reserved for a future GME-style "
                "negative-training change. Keep it false for fair baseline runs."
            )
        if self.hard_negative_mining:
            raise ValueError(
                "hard_negative_mining is reserved for a future data-mining "
                "change. TrainingCollator does not mine or write negatives."
            )

    def _process_payloads(self, payloads: list[dict[str, Any]]) -> list[dict]:
        """Process finalized field payloads with schema-aware fallback rules."""
        if hasattr(self.wrapper, "process_multimodal_batch"):
            return self.wrapper.process_multimodal_batch(payloads)

        texts: list[str] = []
        images: list[Any] = []
        media_metadata: list[dict[str, Any] | None] = []
        for payload in payloads:
            if "media" in payload:
                raise TypeError(
                    f"{type(self.wrapper).__name__} does not support MultiModalInput.media payloads; "
                    "provide a ProcessorWrapper with process_multimodal_batch()"
                )
            texts.append(payload.get("text", "") or " ")
            images.append(payload.get("images", []))
            metadata = payload.get("media_metadata")
            media_metadata.append(metadata if isinstance(metadata, dict) else None)

        return self.wrapper(texts, images, media_metadata=media_metadata)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TrainingCollator:
        """Create collator from configuration dict.


        Args:
            config: Configuration dict. Must include a ``wrapper`` field with a
                ProcessorWrapper instance.

        Returns:
            Initialized TrainingCollator.
        """
        config = config.copy()
        config.pop("type", None)
        # Legacy: ignore adapter config if present
        config.pop("adapter", None)

        instruction_policy = config.pop("instruction_policy", None)
        use_explicit_negatives = _parse_bool_config(
            config.pop("use_explicit_negatives", False),
            field_name="use_explicit_negatives",
        )
        hard_negative_mining = _parse_bool_config(
            config.pop("hard_negative_mining", False),
            field_name="hard_negative_mining",
        )

        wrapper = config.pop("wrapper", None)
        if wrapper is None:
            raise ValueError(
                "'wrapper' must be provided. Create it with "
                "create_processor_wrapper(processor=processor)"
            )

        return cls(
            wrapper=wrapper,
            instruction_policy=instruction_policy,
            use_explicit_negatives=use_explicit_negatives,
            hard_negative_mining=hard_negative_mining,
        )

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process a batch of training samples.


        Args:
            batch: List of samples from Lance dataset. Each sample should have:
                   - query: {"text": str, "media": [...]}
                   - positive: {"text": str, "media": [...]}
                   - negative: {"text": str, "media": [...]}  # Empty text = no negative
                   - dataset_name: str
                   - metadata: dict[str, Any]  # optional runtime metadata

        Returns:
            Dict with ``list[dict | None]`` for each field:
            {
                "query": [
                    {"input_ids": Tensor, "pixel_values": Tensor, ...},  # sample 0
                    {"input_ids": Tensor, "pixel_values": Tensor, ...},  # sample 1
                    ...
                ],
                "positive": [...],
                "negative": [Dict | None, ...],  # None if text is empty
                "metadata": {
                    "ntp_side": ["query", ...],
                    ...
                },
                ...  # Any other fields from batch samples (passthrough)
            }

            Use batch_processor_outputs() to batch before model forward:
            >>> from vlm2emb.data.processors import batch_processor_outputs
            >>> batched_query = batch_processor_outputs(outputs["query"], wrapper)
        """
        outputs: dict[str, Any] = {}

        for field_name in self.CORE_FIELDS:
            payloads: list[dict[str, Any]] = []
            has_content: list[bool] = []  # Track which samples have content

            for sample in batch:
                field_data = sample.get(field_name, {})
                text = field_data.get("text", "") or ""
                rendered_text = render_instruction_text(
                    text,
                    field_name=field_name,
                    instruction_policy=self.instruction_policy,
                )

                # For negatives, empty text means no negative sample is present.
                if field_name == "negative" and not text:
                    has_content.append(False)
                    payloads.append({"text": "", "media": []})  # Placeholder
                else:
                    has_content.append(True)
                    payloads.append(
                        _build_train_field_payload(
                            field_data,
                            sample,
                            field_name=field_name,
                            text=rendered_text or " ",
                        )
                    )

            # Process all samples with the wrapper.
            if field_name == "negative":
                # Only process negative samples that contain usable content.
                valid_payloads = [p for p, h in zip(payloads, has_content, strict=True) if h]

                if valid_payloads:
                    processed = self._process_payloads(valid_payloads)
                    # Reconstruct the original batch shape with None for empty negatives.
                    result: list[dict | None] = []
                    proc_idx = 0
                    for h in has_content:
                        if h:
                            result.append(processed[proc_idx])
                            proc_idx += 1
                        else:
                            result.append(None)
                    outputs[field_name] = result
                else:
                    # All negatives are empty.
                    outputs[field_name] = [None] * len(batch)
            else:
                # Queries and positives are always processed for every sample.
                outputs[field_name] = self._process_payloads(payloads)

        metadata_field_names: set[str] = set()
        for sample in batch:
            metadata = sample.get("metadata")
            if metadata is None:
                continue
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"sample.metadata must be dict when present, got {type(metadata).__name__}"
                )
            metadata_field_names.update(metadata.keys())

        if metadata_field_names:
            outputs["metadata"] = {
                key: [
                    (
                        sample.get("metadata", {}).get(key)
                        if isinstance(sample.get("metadata", {}), dict)
                        else None
                    )
                    for sample in batch
                ]
                for key in sorted(metadata_field_names)
            }

        # Generic metadata passthrough: collect all non-core fields from batch samples
        if batch:
            sample_keys = set().union(*(sample.keys() for sample in batch))
            passthrough_keys = sample_keys - set(self.CORE_FIELDS) - {"metadata"}
            for key in passthrough_keys:
                outputs[key] = [s.get(key) for s in batch]

        return outputs
