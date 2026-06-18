"""Backbone modules for VLM2Emb/BToks embedding pipelines.

This module provides backbone wrappers for various VLM models.
Each backbone converts VLM forward pass to a standardized output format.

Output keys:
    - last_hidden_state: (B, L, D) - Last layer hidden states
    - attention_mask: (B, L) - Attention mask (may be expanded for images)
    - hidden_states: tuple[Tensor] | None - All layer hidden states
"""

import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def _validate_pretrained_source(model_name_or_path: str) -> None:
    """Fail early when an absolute local checkpoint path is not mounted."""
    source_path = Path(model_name_or_path).expanduser()
    if source_path.is_absolute() and not source_path.exists():
        raise FileNotFoundError(
            f"Local checkpoint path does not exist: {source_path}. "
            "Check that the checkpoint is mounted on this node, or use a valid "
            "Hugging Face repo id."
        )


class BackboneBase(nn.Module):
    """Abstract base class for all backbone modules.

    Used for type identification in PEFT modules_to_save inference.
    All backbone implementations (Qwen2VLBackbone, Qwen3VLBackbone, etc.)
    must inherit from this class.
    """

    def build_continuation_position_ids(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        start_positions: Tensor,
        cache_position: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Build RoPE positions for continuation from compressed cache."""
        rope_owner = self._get_rope_owner()
        if rope_owner is not None and hasattr(rope_owner, "get_rope_index"):
            position_ids, _ = rope_owner.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            return position_ids + start_positions.view(1, -1, 1)

        offsets = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        return offsets + start_positions.unsqueeze(1)

    def _get_rope_owner(self) -> Any | None:
        """Return the wrapped module that owns ``get_rope_index`` when present."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        if hasattr(model, "get_rope_index"):
            return model
        return getattr(model, "model", None)


class Qwen2VLBackbone(BackboneBase):
    """Qwen2-VL backbone module for embedding pipeline.

    This module wraps Qwen2VLForConditionalGeneration to provide a standardized
    interface for the VLM2Emb pipeline. It extracts hidden states from the
    language model for downstream pooling.

    Attributes:
        model: Wrapped Qwen2-VL generation model.
        hidden_size: Hidden dimension used by downstream pooling modules.

    Example:
        >>> backbone = Qwen2VLBackbone.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> output = backbone(input_ids=tokens, pixel_values=images)
        >>> hidden_states = output["last_hidden_state"]  # (B, L, D)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int | None = None,
    ):
        """Initialize Qwen2VLBackbone around a preloaded model instance.

        Args:
            model: Preloaded Qwen2-VL model instance.
            hidden_size: Hidden dimension. If omitted, it is inferred from
                ``model.config.hidden_size``.
        """
        super().__init__()
        self.model: Any = model
        model_config = cast(Any, model).config
        self._hidden_size = hidden_size or model_config.hidden_size

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device_map: str | None = None,  # None for distributed training, "auto" for single GPU
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "Qwen2VLBackbone":
        """Load Qwen2VL model from pretrained weights.

        Note: PEFT/LoRA should be applied externally to the retrieval model,
        not inside the backbone. Use train.py or configuration to manage PEFT.

        Args:
            model_name_or_path: HuggingFace model name or local path
            dtype: Model dtype ("float32", "float16", "bfloat16")
            attn_implementation: Attention implementation
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading

        Returns:
            Initialized Qwen2VLBackbone instance
        """
        _validate_pretrained_source(model_name_or_path)

        from transformers import Qwen2VLForConditionalGeneration

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if isinstance(dtype, str):
            dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info(f"Loading Qwen2VL model from {model_name_or_path}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return cls(model=model)

    @classmethod
    def from_config(cls, config: dict) -> "Qwen2VLBackbone":
        """Create from configuration dict (dual-mode).

        Supports two modes:
        - **Structure mode**: config has ``backbone_config`` -> build structure only
          (random weights), used by ``VLM2Emb.from_pretrained``.
        - **Load mode**: config has ``model_name_or_path`` -> load pretrained weights,
          used by ``create_model``.

        Args:
            config: Configuration dict with either ``backbone_config`` or
                    ``model_name_or_path``.

        Returns:
            Initialized Qwen2VLBackbone instance
        """
        config = config.copy()
        config.pop("type", None)
        if "backbone_config" in config:
            from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

            hf_config = Qwen2VLConfig(**config["backbone_config"])
            model = Qwen2VLForConditionalGeneration(hf_config)
            return cls(model=model)
        return cls.from_pretrained(**config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass through the wrapped Qwen2-VL model.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions
            pixel_values_videos: Video pixel values
            video_grid_thw: Video grid dimensions
            output_hidden_states: Whether to return all layer hidden states
            **kwargs: Additional arguments (passed through)
                - labels: Tensor | None - Labels for generation loss computation
                - return_cache: bool - If True, return past_key_values for caching
                - cache_position: Tensor | None - Position indices for KV cache continuation

        Returns:
            Dict with keys:
                - last_hidden_state: (B, L, D)
                - attention_mask: (B, L) - may be expanded
                - hidden_states: tuple | None
                - loss: Tensor | None - Generation loss (if labels provided)
                - past_key_values: tuple | None - KV cache (if return_cache=True)
                - input_ids, pixel_values, etc. (passthrough)
        """
        # Extract generation-related kwargs before passing remaining features through.
        labels = kwargs.pop("labels", None)
        return_cache = kwargs.pop("return_cache", False)
        past_key_values = kwargs.pop("past_key_values", None)
        cache_position = kwargs.pop("cache_position", None)
        position_ids = kwargs.pop("position_ids", None)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }

        # Enable KV cache when requested or when an existing cache is provided.
        if return_cache or past_key_values is not None:
            model_inputs["use_cache"] = True

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

        if pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
            if video_grid_thw is not None:
                model_inputs["video_grid_thw"] = video_grid_thw

        # Qwen2VLModel already performs multimodal embedding injection and returns
        # last_hidden_state directly. Avoid requesting the full hidden_states tuple
        # unless the caller explicitly needs it.
        outputs = self.model.model(**model_inputs)
        last_hidden_states = outputs.last_hidden_state

        hidden_states = outputs.hidden_states if output_hidden_states else None

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Return a feature dict compatible with downstream pooling modules.
        return_dict = {
            "last_hidden_state": last_hidden_states,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        # Compute generation loss only when labels are explicitly provided.
        if labels is not None:
            loss_outputs = self.model(
                **model_inputs,
                labels=labels,
                return_dict=True,
            )
            if loss_outputs.loss is not None:
                return_dict["loss"] = loss_outputs.loss

        # Preserve KV cache only when the caller requested cache output.
        if return_cache and outputs.past_key_values is not None:
            return_dict["past_key_values"] = outputs.past_key_values

        return return_dict

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension size."""
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return next(self.model.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Enable gradient checkpointing."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()


class Qwen3VLBackbone(BackboneBase):
    """Qwen3-VL backbone module for embedding pipeline.

    This module wraps Qwen3VLForConditionalGeneration to provide a standardized
    interface for the VLM2Emb pipeline. It extracts hidden states from the
    language model for downstream pooling.

    Attributes:
        model: Wrapped Qwen3-VL generation model.
        hidden_size: Hidden dimension used by downstream pooling modules.

    Example:
        >>> backbone = Qwen3VLBackbone.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        >>> output = backbone(input_ids=tokens, pixel_values=images)
        >>> hidden_states = output["last_hidden_state"]  # (B, L, D)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int | None = None,
    ):
        """Initialize Qwen3VLBackbone around a preloaded model instance.

        Args:
            model: Preloaded Qwen3-VL model instance.
            hidden_size: Hidden dimension. If omitted, it is inferred from the
                text config and then the vision config.
        """
        super().__init__()
        self.model: Any = model
        self._hidden_size = (
            hidden_size
            or cast(Any, model).config.text_config.hidden_size
            or cast(Any, model).config.vision_config.hidden_size
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device_map: str | None = None,  # None for distributed training, "auto" for single GPU
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "Qwen3VLBackbone":
        """Load Qwen3VL model from pretrained weights.

        Note: PEFT/LoRA should be applied externally to the retrieval model,
        not inside the backbone. Use train.py or configuration to manage PEFT.

        Args:
            model_name_or_path: HuggingFace model name or local path
            dtype: Model dtype ("float32", "float16", "bfloat16")
            attn_implementation: Attention implementation
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading

        Returns:
            Initialized Qwen3VLBackbone instance
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if isinstance(dtype, str):
            dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info(f"Loading Qwen3VL model from {model_name_or_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return cls(model=model)

    @classmethod
    def from_config(cls, config: dict) -> "Qwen3VLBackbone":
        """Create from configuration dict (dual-mode).

        Supports two modes:
        - **Structure mode**: config has ``backbone_config`` -> build structure only
          (random weights), used by ``VLM2Emb.from_pretrained``.
        - **Load mode**: config has ``model_name_or_path`` -> load pretrained weights,
          used by ``create_model``.

        Args:
            config: Configuration dict with either ``backbone_config`` or
                    ``model_name_or_path``.

        Returns:
            Initialized Qwen3VLBackbone instance
        """
        config = config.copy()
        config.pop("type", None)
        if "backbone_config" in config:
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )

            hf_config = Qwen3VLConfig(**config["backbone_config"])
            model = Qwen3VLForConditionalGeneration(hf_config)
            return cls(model=model)
        return cls.from_pretrained(**config)

    def build_continuation_position_ids(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        start_positions: Tensor,
        cache_position: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Build Qwen3-VL continuation positions with separate FA text ids."""
        mrope_position_ids = super().build_continuation_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            cache_position=cache_position,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        if mrope_position_ids.ndim != 3:
            return mrope_position_ids

        if cache_position is None:
            text_position_ids = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            text_position_ids = cache_position.to(device=input_ids.device, dtype=torch.long)
        text_position_ids = text_position_ids.view(1, 1, -1).expand(
            1,
            input_ids.shape[0],
            -1,
        )
        return torch.cat([text_position_ids, mrope_position_ids], dim=0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass through the wrapped Qwen3-VL model.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)
            pixel_values: Image features from processor
            image_grid_thw: Image grid dimensions (Qwen3-VL specific)
            pixel_values_videos: Video features
            video_grid_thw: Video grid dimensions
            output_hidden_states: Whether to return all layer hidden states
            **kwargs: Additional arguments (passed through)
                - labels: Tensor | None - Labels for generation loss computation
                - return_cache: bool - If True, return past_key_values for caching
                - cache_position: Tensor | None - Position indices for KV cache continuation

        Returns:
            Dict with keys:
                - last_hidden_state: (B, L, D)
                - attention_mask: (B, L) - may be expanded
                - hidden_states: tuple | None
                - loss: Tensor | None - Generation loss (if labels provided)
                - past_key_values: tuple | None - KV cache (if return_cache=True)
                - input_ids, pixel_values, etc. (passthrough)
        """
        if output_hidden_states:
            raise NotImplementedError(
                "Qwen3VLBackbone does not support output_hidden_states yet. "
                "Use last_hidden_state instead."
            )

        # Extract generation-related kwargs before passing remaining features through.
        labels = kwargs.pop("labels", None)
        return_cache = kwargs.pop("return_cache", False)
        past_key_values = kwargs.pop("past_key_values", None)
        cache_position = kwargs.pop("cache_position", None)
        position_ids = kwargs.pop("position_ids", None)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "return_dict": True,
        }

        # Enable KV cache when requested or when an existing cache is provided.
        if return_cache or past_key_values is not None:
            model_inputs["use_cache"] = True

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

        if pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
            if video_grid_thw is not None:
                model_inputs["video_grid_thw"] = video_grid_thw

        # Use the inner multimodal model for embedding extraction.
        # Qwen3VLForConditionalGeneration returns generation-oriented outputs,
        # while Qwen3VLModel returns last_hidden_state for pooling.
        outputs = self.model.model(**model_inputs)
        last_hidden_states = outputs.last_hidden_state
        hidden_states = None

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Return a feature dict compatible with downstream pooling modules.
        return_dict = {
            "last_hidden_state": last_hidden_states,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        # Loss computation still uses the top-level causal LM when explicitly requested.
        if labels is not None:
            loss_outputs = self.model(
                **model_inputs,
                labels=labels,
                return_dict=True,
            )
            if loss_outputs.loss is not None:
                return_dict["loss"] = loss_outputs.loss

        # Preserve KV cache only when the caller requested cache output.
        if return_cache and outputs.past_key_values is not None:
            return_dict["past_key_values"] = outputs.past_key_values

        return return_dict

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension size."""
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return next(self.model.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Enable gradient checkpointing."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()


class Qwen2_5VLBackbone(BackboneBase):
    """Qwen2.5-VL backbone module for embedding pipeline."""

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int | None = None,
    ):
        super().__init__()
        self.model: Any = model
        self._hidden_size = (
            hidden_size
            or getattr(cast(Any, model).config, "hidden_size", None)
            or cast(Any, model).config.text_config.hidden_size
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device_map: str | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "Qwen2_5VLBackbone":
        from transformers import Qwen2_5_VLForConditionalGeneration

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if isinstance(dtype, str):
            dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info(f"Loading Qwen2.5-VL model from {model_name_or_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(model=model)

    @classmethod
    def from_config(cls, config: dict) -> "Qwen2_5VLBackbone":
        config = config.copy()
        config.pop("type", None)
        if "backbone_config" in config:
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
            )

            hf_config = Qwen2_5_VLConfig(**config["backbone_config"])
            model = Qwen2_5_VLForConditionalGeneration(hf_config)
            return cls(model=model)
        return cls.from_pretrained(**config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        labels = kwargs.pop("labels", None)
        return_cache = kwargs.pop("return_cache", False)
        past_key_values = kwargs.pop("past_key_values", None)
        cache_position = kwargs.pop("cache_position", None)
        position_ids = kwargs.pop("position_ids", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }

        if return_cache or past_key_values is not None:
            model_inputs["use_cache"] = True

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

        if pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
            if video_grid_thw is not None:
                model_inputs["video_grid_thw"] = video_grid_thw

        if second_per_grid_ts is not None:
            model_inputs["second_per_grid_ts"] = second_per_grid_ts

        outputs = self.model.model(**model_inputs)
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states if output_hidden_states else None

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        return_dict = {
            "last_hidden_state": last_hidden_states,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
        }

        if labels is not None:
            loss_outputs = self.model(
                **model_inputs,
                labels=labels,
                return_dict=True,
            )
            if loss_outputs.loss is not None:
                return_dict["loss"] = loss_outputs.loss

        if return_cache and outputs.past_key_values is not None:
            return_dict["past_key_values"] = outputs.past_key_values

        return return_dict

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()
