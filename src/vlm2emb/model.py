"""BToks/VLM2Vec model implementation with module pipeline architecture.

This module implements the retrieval model following the sentence-transformers
design pattern. The model is composed of modules that execute sequentially and
pass a features dictionary between stages.

Example:
    >>> from vlm2emb import create_model
    >>> from vlm2emb.config import load_config
    >>>
    >>> # From YAML config (recommended for training)
    >>> config = load_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
    >>> model = create_model(config.model)
    >>>
    >>> # From pretrained (for inference)
    >>> model = BToks.from_pretrained("public-model-or-checkpoint")
"""

import logging
import os
import re
from typing import Any, TypeVar, cast

import torch.nn as nn
from omegaconf import OmegaConf
from torch import Tensor
from transformers import AutoConfig as HFAutoConfig
from transformers import AutoModel as HFAutoModel
from transformers import PretrainedConfig, PreTrainedModel

from vlm2emb.auto import AutoModule

logger = logging.getLogger(__name__)

TModule = TypeVar("TModule", bound=nn.Module)

CONFIG_LOAD_KWARG_NAMES = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
    "use_auth_token",
}


# ============================================================================
# Model Factory
# ============================================================================


def create_model(model_config: Any | dict) -> PreTrainedModel:
    """Create model from model configuration subtree.

    This function provides a configuration-driven way to create models.
    The model type is determined by the `type` field in the model config.
    Uses HuggingFace AutoConfig/AutoModel for unified ecosystem integration.

    Args:
        model_config: Model configuration subtree (config.model).

    Returns:
        Model instance.

    Raises:
        ValueError: If model type is not registered with HuggingFace AutoConfig.

    Example:
        >>> from vlm2emb import create_model
        >>> from vlm2emb.config import load_config
        >>>
        >>> config = load_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
        >>> model = create_model(config.model)
    """
    # Convert to dict if needed.
    if not isinstance(model_config, dict):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    # Copy to avoid mutating the original config.
    model_config = model_config.copy()

    # Get and remove type field, convert to lowercase for HF compatibility.
    model_type = model_config.pop("type", "vlm2emb").lower()

    # Use HuggingFace AutoConfig to create config instance.
    hf_config = HFAutoConfig.for_model(model_type, **model_config)

    # Use HuggingFace AutoModel to create model from config.
    model = HFAutoModel.from_config(hf_config)

    # Post-build: add backbone_config alongside model_name_or_path.
    # Both are kept so save_pretrained can choose which to write based on save_base_weights.
    if hasattr(model, "config") and hasattr(model.config, "modules"):
        for name, mod_cfg in model.config.modules.items():
            if isinstance(mod_cfg, dict) and "model_name_or_path" in mod_cfg:
                backbone = model._modules_dict[name]
                if hasattr(backbone, "model") and hasattr(backbone.model, "config"):
                    mod_cfg["backbone_config"] = backbone.model.config.to_dict()

    return model


def _extract_config_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract kwargs that are relevant when loading Hugging Face configs."""
    return {k: kwargs[k] for k in CONFIG_LOAD_KWARG_NAMES if k in kwargs}


def _resolve_backbone_source_path(model_name_or_path: str) -> str:
    """Resolve historical malformed local-like backbone paths to HF repo IDs."""
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    basename = os.path.basename(model_name_or_path.rstrip("/"))
    if basename.startswith("Qwen") and "/" not in basename:
        return f"Qwen/{basename}"
    return model_name_or_path


def _load_backbone_config_snapshot(module_config: dict[str, Any]) -> dict[str, Any]:
    """Load a serialized backbone config snapshot from model_name_or_path."""
    load_kwargs = {}
    if "revision" in module_config:
        load_kwargs["revision"] = module_config["revision"]
    if "subfolder" in module_config:
        load_kwargs["subfolder"] = module_config["subfolder"]
    if "token" in module_config:
        load_kwargs["token"] = module_config["token"]
    if "use_auth_token" in module_config:
        load_kwargs["use_auth_token"] = module_config["use_auth_token"]
    if "trust_remote_code" in module_config:
        load_kwargs["trust_remote_code"] = module_config["trust_remote_code"]

    hf_config = HFAutoConfig.from_pretrained(
        _resolve_backbone_source_path(module_config["model_name_or_path"]),
        **load_kwargs,
    )
    return hf_config.to_dict()


def _ensure_artifact_backbone_configs(
    modules_config: dict[str, dict[str, Any]],
    modules_dict: nn.ModuleDict | None = None,
) -> None:
    """Ensure artifact configs store backbone_config for referenced modules."""
    for name, mod_cfg in modules_config.items():
        if not isinstance(mod_cfg, dict):
            continue
        if "model_name_or_path" not in mod_cfg or "backbone_config" in mod_cfg:
            continue

        runtime_module = None
        if modules_dict is not None and name in modules_dict:
            runtime_module = modules_dict[name]

        if (
            runtime_module is not None
            and hasattr(runtime_module, "model")
            and hasattr(runtime_module.model, "config")
        ):
            runtime_model = cast(Any, runtime_module).model
            mod_cfg["backbone_config"] = runtime_model.config.to_dict()
        else:
            mod_cfg["backbone_config"] = _load_backbone_config_snapshot(mod_cfg)


def _prepare_artifact_config_for_load(
    cls,
    pretrained_model_name_or_path: str | os.PathLike,
    kwargs: dict[str, Any],
) -> tuple[PretrainedConfig, dict[str, Any]]:
    """Prepare artifact config so reference modules build as skeletons."""
    load_kwargs = kwargs.copy()
    config_obj = load_kwargs.get("config")
    if config_obj is None:
        config_load_kwargs = _extract_config_load_kwargs(load_kwargs)
        config_obj = cls.config_class.from_pretrained(
            pretrained_model_name_or_path,
            **config_load_kwargs,
        )
    elif isinstance(config_obj, dict):
        config_obj = cls.config_class(**config_obj)

    if not isinstance(config_obj, PretrainedConfig):
        raise TypeError(
            f"config must be a dict or PretrainedConfig, got {type(config_obj).__name__}"
        )

    if hasattr(config_obj, "modules"):
        _ensure_artifact_backbone_configs(config_obj.modules)

    load_kwargs["config"] = config_obj
    return config_obj, load_kwargs


def _infer_backbone_path(model: "VLM2Emb") -> str | None:
    """Infer backbone model_name_or_path from model config modules.

    Iterates model.config.modules looking for entries with 'model_name_or_path'.
    Returns the first found, or None.

    Args:
        model: Retrieval model instance.

    Returns:
        The backbone model_name_or_path string, or None if not found.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "modules"):
        return None
    for mod_cfg in model.config.modules.values():
        if isinstance(mod_cfg, dict) and "model_name_or_path" in mod_cfg:
            return mod_cfg["model_name_or_path"]
    return None


def create_model_and_processor(
    config: dict,
) -> tuple["VLM2Emb", Any]:
    """Create model and its associated processing class, with setup binding.

    Unifies the model + processor creation logic used by both train.py and eval.

    Steps:
    1. create_model(config["model"])
    2. Load processor:
       a. If config has "processor" key -> AutoProcessorWrapper.from_config(config["processor"]).processor
       b. Else -> infer from backbone model_name_or_path -> AutoProcessor.from_pretrained
       c. Neither -> processing_class = None
    3. If processing_class is not None -> model.setup(processor=processing_class)
    4. Return (model, processing_class)

    Args:
        config: Full configuration dict (already converted from OmegaConf).

    Returns:
        (model, processing_class) where processing_class may be Processor, Tokenizer, or None.
    """
    # Step 1: create the model from the resolved config.
    model = cast(VLM2Emb, create_model(config["model"]))
    logger.info("Model created: %s", type(model).__name__)

    # Step 2: load the processor from explicit config or the backbone source.
    processing_class = None

    if "processor" in config and config["processor"] is not None:
        # Explicit processor config via AutoProcessorWrapper registry.
        from vlm2emb.auto import AutoProcessorWrapper

        logger.info("Loading processor from config (type=%s)", config["processor"].get("type"))
        wrapper = AutoProcessorWrapper.from_config(config["processor"])
        processing_class = wrapper.processor
        logger.info("Processor loaded: %s", type(processing_class).__name__)
    else:
        # Fallback: infer the processor from backbone model_name_or_path.
        backbone_path = _infer_backbone_path(model)
        if backbone_path:
            from transformers import AutoProcessor

            logger.info("Loading processor from backbone path: %s", backbone_path)
            processing_class = AutoProcessor.from_pretrained(
                backbone_path,
                trust_remote_code=True,
            )
            logger.info("Processor loaded: %s", type(processing_class).__name__)
        else:
            # No processor is available for this config.
            logger.info("No processor config or backbone path found; processing_class is None")

    # Step 3: bind the processor to interested modules via setup hooks.
    if processing_class is not None:
        model.setup(processor=processing_class)

    # Step 4: return both runtime objects.
    return model, processing_class


# ============================================================================
# Public Model Configuration
# ============================================================================


class VLM2EmbConfig(PretrainedConfig):
    """Configuration class for the public retrieval model.

    This configuration holds all parameters needed to instantiate a retrieval model
    with a pipeline of modules. The model architecture is defined by `modules`,
    a list of module configurations that are processed sequentially.

    Args:
        modules: List of module configuration dicts. Each dict must have
            a 'type' key specifying the module class, plus module-specific parameters.

        normalize_embeddings: Whether to normalize embeddings by default.

    Example:
        >>> config = VLM2EmbConfig(
        ...     modules=[
        ...         {"type": "Qwen2VLTransformer", "model_name_or_path": "Qwen/Qwen2-VL-7B"},
        ...         {"type": "Pooling", "pooling_mode": "lasttoken"},
        ...         {"type": "Normalize"},
        ...     ],
        ... )
    """

    model_type = "vlm2emb"

    def __init__(
        self,
        modules: list[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        """Initialize VLM2EmbConfig.

        Args:
            modules: Module configs as dict (name -> config) or legacy list format.
            normalize_embeddings: Whether to normalize embeddings by default.
        """
        super().__init__(**kwargs)
        # Normalize: accept both dict and legacy list format, store as dict
        if modules is None:
            self.modules = {}
        elif isinstance(modules, dict):
            self.modules = modules
        elif isinstance(modules, list):
            # Legacy list format: auto-assign names from type field
            self.modules = self._list_to_dict(modules)
        else:
            raise TypeError(f"modules must be dict or list, got {type(modules).__name__}")

        # Preserve module order: HF config serializes with sort_keys=True,
        # which destroys dict insertion order. Store order explicitly.
        if "_module_order" not in kwargs:
            self._module_order = list(self.modules.keys())
        else:
            self._module_order = kwargs["_module_order"]
            # Reorder modules dict to match stored order
            reordered = {k: self.modules[k] for k in self._module_order if k in self.modules}
            # Append any modules not in _module_order (forward compat)
            for k in self.modules:
                if k not in reordered:
                    reordered[k] = self.modules[k]
            self.modules = reordered

        self.normalize_embeddings = normalize_embeddings

    @staticmethod
    def _list_to_dict(modules_list: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Convert legacy list format to dict format with auto-generated names."""
        result = {}
        name_counts: dict[str, int] = {}
        for mod_cfg in modules_list:
            if not isinstance(mod_cfg, dict):
                raise ValueError(f"Each module config must be a dict, got {type(mod_cfg).__name__}")
            # Derive name from type, lowercased
            mod_type = mod_cfg.get("type", "module")
            # Convert CamelCase to snake_case for name
            base_name = mod_type[0].lower() + mod_type[1:]
            base_name = "".join(
                f"_{c.lower()}" if c.isupper() else c for c in base_name
            )
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            name = base_name if count == 0 else f"{base_name}_{count}"
            result[name] = mod_cfg
        return result


class BToksConfig(VLM2EmbConfig):
    """Public BToks config alias for the VLM2Emb pipeline runtime."""

    model_type = "btoks"


# ============================================================================
# Public retrieval model
# ============================================================================


class VLM2Emb(PreTrainedModel):
    """Public retrieval model implementation.

    This model implements a modular pipeline architecture inspired by sentence-transformers.
    It consists of a sequence of modules (Transformer, Pooling, Dense, Normalize, etc.)
    that transform inputs into fixed-size embedding vectors.

    Different model styles are achieved through different module configurations:

    - VLM2Vec style: [Transformer, LastTokenPooling, Normalize]
    - BToks style: [Transformer, BottleneckTokens, Pooling, Normalize]

    Architecture:
        Input -> Module_0 (Transformer) -> Module_1 (Pooling) -> ... -> Embeddings

    The modules communicate through a `features` dictionary that is passed through
    the pipeline. Standard keys include:
        - input_ids: Token IDs (B, L)
        - attention_mask: Attention mask (B, L)
        - pixel_values: Image features
        - token_embeddings: Transformer output (B, L, D)
        - embeddings: Final embedding (B, D)

    Example:
        >>> from vlm2emb import create_model
        >>> from vlm2emb.config import load_config
        >>>
        >>> config = load_config("configs/presets/vlm2vec_qwen2vl_2b.yaml")
        >>> model = create_model(config.model)
        >>>
        >>> # Training forward
        >>> features = model(input_ids=tokens, attention_mask=mask, pixel_values=images)
        >>> embeddings = features["embeddings"]
        >>>
        >>> # Inference: Use forward() with preprocessed inputs
        >>> inputs = processor(text=["query"], images=[image], return_tensors="pt")
        >>> inputs = {k: v.to(model.device) for k, v in inputs.items()}
        >>> features = model(**inputs)
        >>> embeddings = features["embeddings"]
    """

    config_class = VLM2EmbConfig
    base_model_prefix = "vlm2emb"
    supports_gradient_checkpointing = True

    def __init__(self, config: VLM2EmbConfig):
        """Initialize the public retrieval model.

        Args:
            config: Model configuration with modules dict (or legacy list)
        """
        super().__init__(config)

        # Reentrant guard for forward_fn injection
        self._in_forward = False

        # Build the module pipeline from the stored config order.
        self._modules_dict = nn.ModuleDict()
        self._build_modules(config.modules)

        # Reference artifacts omit pretrained module weights from the local
        # state_dict. Keep ignore patterns so skeleton builds do not emit noisy
        # missing-key warnings before post-load re-materialization.
        ignore_patterns = []
        for name, mod_cfg in config.modules.items():
            if isinstance(mod_cfg, dict) and "model_name_or_path" in mod_cfg:
                ignore_patterns.append(rf"_modules_dict\.{re.escape(name)}\.")
        if ignore_patterns:
            self._keys_to_ignore_on_load_missing = ignore_patterns

        # Register hook for legacy checkpoint migration
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        # Initialize weights as required by PreTrainedModel.
        self.post_init()

    def _build_modules(self, modules_config: dict[str, dict[str, Any]]) -> None:
        """Build modules from configuration dict.

        Args:
            modules_config: Dict of name -> module configuration
        """
        for name, module_config in modules_config.items():
            if not isinstance(module_config, dict):
                raise ValueError(f"modules[{name!r}] must be a dict, got {type(module_config).__name__}")

            module = AutoModule.from_config(module_config)
            self._modules_dict[name] = module
            logger.debug(f"Built module {name!r}: {module_config.get('type', 'unknown')}")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        forward_fn: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass through module pipeline, with optional forward_fn injection.

        When forward_fn is None, calls self.encode() to run the default pipeline.
        When forward_fn is provided, it completely replaces the default pipeline.

        Args:
            input_ids: Token IDs from tokenizer (B, L)
            attention_mask: Attention mask (B, L)
            pixel_values: Image features from processor
            image_grid_thw: Image grid dimensions (Qwen2-VL)
            pixel_values_videos: Video features (Qwen2-VL)
            video_grid_thw: Video grid dimensions (Qwen2-VL)
            forward_fn: Optional callable (model, **features) -> dict.
                        When provided, replaces the default encode pipeline.
            **kwargs: Additional model-specific inputs

        Returns:
            Features dict with at least 'embeddings' key containing
            the final embeddings of shape (B, D).
        """
        if self._in_forward:
            raise RuntimeError(
                "Reentrant forward() detected. "
                "Use model.encode() inside forward_fn instead of model() or model.forward()."
            )
        self._in_forward = True
        try:
            # Initialize features dict, filter None values
            features: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                **kwargs,
            }
            features = {k: v for k, v in features.items() if v is not None}

            if forward_fn is not None:
                return forward_fn(self, **features)

            return self.encode(**features)
        finally:
            self._in_forward = False

    def encode(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute the default module pipeline.

        Runs each module in _modules_dict insertion order, merging outputs
        into the features dict. This is the method forward_fn should call
        instead of model.forward() to avoid reentrant guard.

        Args:
            input_ids: Token IDs from tokenizer (B, L)
            attention_mask: Attention mask (B, L)
            pixel_values: Image features from processor
            image_grid_thw: Image grid dimensions (Qwen2-VL)
            pixel_values_videos: Video features (Qwen2-VL)
            video_grid_thw: Video grid dimensions (Qwen2-VL)
            **kwargs: Additional model-specific inputs

        Returns:
            Features dict with at least 'embeddings' key.
        """
        features: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            **kwargs,
        }
        features = {k: v for k, v in features.items() if v is not None}

        for _key, module in self._modules_dict.items():
            output = module(**features)
            features.update(output)

        return features

    # @property
    # def device(self) -> torch.device:
    #     """Get model device."""
    #     return next(self.parameters()).device

    # def find_module(self, module_type: type[TModule]) -> TModule | None:
    #     """Find first module instance by type.

    #     .. deprecated::
    #         Use ``model._modules_dict["name"]`` for direct access by name.

    #     Args:
    #         module_type: Module class/type to match.

    #     Returns:
    #         First matched module instance, or None if not found.
    #     """
    #     import warnings
    #     warnings.warn(
    #         "find_module() is deprecated. Use model._modules_dict['name'] instead.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     for module in self._modules_dict.values():
    #         if isinstance(module, module_type):
    #             return module
    #     return None

    def find_backbone(self) -> nn.Module | None:
        """Find the first BackboneBase module in the pipeline.

        .. deprecated::
            Use ``model._modules_dict["backbone_name"]`` for direct access.
        """
        import warnings
        warnings.warn(
            "find_backbone() is deprecated. Use model._modules_dict['backbone_name'] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from vlm2emb.modules.backbone import BackboneBase

        for module in self._modules_dict.values():
            if isinstance(module, BackboneBase):
                return module
        return None

    def infer_modules_to_save(self) -> list[str]:
        """Infer modules_to_save by scanning _modules_dict.

        For each non-backbone module, calls modules_to_save() if implemented
        to get fine-grained sub-module paths. Falls back to the whole module
        if modules_to_save() is not implemented but the module has parameters.

        Returns:
            List of module paths for modules_to_save
        """
        from vlm2emb.modules.backbone import BackboneBase

        result = []
        for name, module in self._modules_dict.items():
            if isinstance(module, BackboneBase):
                continue  # backbone — handled by LoRA target_modules
            # Check if module declares fine-grained sub-modules
            getter = getattr(module, "modules_to_save", None)
            if callable(getter):
                for sub_name in cast(list[str], getter()):
                    result.append(f"_modules_dict.{name}.{sub_name}")
            elif sum(p.numel() for p in module.parameters()) > 0:
                result.append(f"_modules_dict.{name}")
        return result

    def get_input_embeddings(self) -> nn.Module | None:
        """Return input embeddings from the first backbone module, as required by PEFT."""
        from vlm2emb.modules.backbone import BackboneBase

        for module in self._modules_dict.values():
            if isinstance(module, BackboneBase):
                return cast(Any, module).model.get_input_embeddings()
        return None

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings on the first backbone module, as required by PEFT."""
        from vlm2emb.modules.backbone import BackboneBase

        for module in self._modules_dict.values():
            if isinstance(module, BackboneBase):
                cast(Any, module).model.set_input_embeddings(value)
                return

    def get_module(self, index: int | str) -> nn.Module:
        """Get module by index or name.

        Args:
            index: Module index (int) or name (str) in pipeline

        Returns:
            Module at the specified index/name
        """
        if isinstance(index, str):
            return self._modules_dict[index]
        return list(self._modules_dict.values())[index]

    # def __len__(self) -> int:
    #     """Return number of modules in pipeline."""
    #     return len(self._modules_dict)

    # def __iter__(self):
    #     """Iterate over modules."""
    #     return iter(self._modules_dict.values())

    def get_trainable_token_indices(self) -> list[int] | None:
        """Collect trainable token indices from all modules.

        Iterates through _modules_dict and calls get_trainable_token_indices()
        on any module that implements this protocol method. Results are merged
        into a single deduplicated list.

        Returns:
            List of token indices, or None if no module provides any.
        """
        all_indices: list[int] = []
        for module in self._modules_dict.values():
            getter = getattr(module, "get_trainable_token_indices", None)
            if getter is not None:
                indices = getter()
                if indices:
                    all_indices.extend(indices)
        if not all_indices:
            return None
        # Deduplicate while preserving order
        seen: set[int] = set()
        unique: list[int] = []
        for idx in all_indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        return unique

    def setup(self, **components) -> None:
        """Pass components to modules that declare interest via on_<name> hooks.

        Currently supported components:
        - processor: HuggingFace processor or tokenizer instance

        Modules opt in by implementing ``on_<component_name>(self, model, value)``.
        For example, ``BToksTokenInjector.on_processor(self, model, processor)``
        adds special tokens and resizes embeddings.

        Args:
            **components: Named components to pass to modules.
        """
        for name, module in self._modules_dict.items():
            for comp_name, comp_value in components.items():
                hook = getattr(module, f"on_{comp_name}", None)
                if hook is not None:
                    logger.info(f"Setting up module {name!r} with {comp_name}")
                    hook(self, comp_value)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *args,
        save_base_weights: bool = True,
        **kwargs,
    ) -> None:
        """Save model with optional reference mode.

        Args:
            save_directory: Directory to save model files.
            save_base_weights: If True (default), save all weights including
                pretrained modules (full mode). If False, exclude pretrained
                module weights and keep both backbone_config and model_name_or_path
                in config (reference mode).
            *args, **kwargs: Passed to PreTrainedModel.save_pretrained.
        """
        _ensure_artifact_backbone_configs(self.config.modules, self._modules_dict)

        # Find pretrained module names, identified by model_name_or_path in config.
        pretrained_names = []
        for name, mod_cfg in self.config.modules.items():
            if isinstance(mod_cfg, dict) and "model_name_or_path" in mod_cfg:
                pretrained_names.append(name)

        if not pretrained_names or save_base_weights:
            # Full mode saves all weights and stores backbone_config in the config.
            saved_fields = {}
            for name in pretrained_names:
                mod_cfg = self.config.modules[name]
                saved_fields[name] = mod_cfg.pop("model_name_or_path", None)

            try:
                super().save_pretrained(save_directory, *args, **kwargs)
            finally:
                # Restore model_name_or_path in memory after writing the artifact.
                for name, val in saved_fields.items():
                    if val is not None:
                        self.config.modules[name]["model_name_or_path"] = val
        else:
            # Reference mode excludes pretrained module weights while keeping both
            # backbone_config (structure) and model_name_or_path (load source) in
            # the artifact config.
            prefixes = tuple(f"_modules_dict.{name}." for name in pretrained_names)
            full_sd = kwargs.pop("state_dict", None)
            if full_sd is None:
                full_sd = self.state_dict()
            custom_sd = {k: v for k, v in full_sd.items() if not k.startswith(prefixes)}

            super().save_pretrained(save_directory, *args, state_dict=custom_sd, **kwargs)

            # If custom_sd is empty (no custom-layer weights), HF won't write a
            # weights file, but from_pretrained requires one. Write an empty safetensors.
            if not custom_sd:
                weights_path = os.path.join(save_directory, "model.safetensors")
                if not os.path.exists(weights_path):
                    from safetensors.torch import save_file

                    save_file({}, weights_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model, re-materializing pretrained modules in reference mode.

        Also handles legacy checkpoint migration from _modules_list.{idx} to
        _modules_dict.{name} key format.
        """
        _prepared_config, load_kwargs = _prepare_artifact_config_for_load(
            cls,
            pretrained_model_name_or_path,
            kwargs,
        )
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **load_kwargs)

        # Re-materialize pretrained modules that were saved in reference mode.
        # Reference artifacts keep both backbone_config (for skeleton build)
        # and model_name_or_path (for a single post-load materialization).
        for name, mod_cfg in model.config.modules.items():
            if (
                isinstance(mod_cfg, dict)
                and "model_name_or_path" in mod_cfg
                and "backbone_config" in mod_cfg
            ):
                # Reference mode: reload this module exactly once from source.
                # Build stage already used backbone_config to avoid eager loading.
                logger.info(
                    f"Reference mode: reloading module {name!r} from {mod_cfg['model_name_or_path']}"
                )
                load_cfg = mod_cfg.copy()
                load_cfg.pop("backbone_config", None)
                load_cfg["model_name_or_path"] = _resolve_backbone_source_path(
                    load_cfg["model_name_or_path"]
                )
                reloaded = AutoModule.from_config(load_cfg)
                model._modules_dict[name] = reloaded

        # Auto-setup: if any module has on_processor hook, load processor and call setup.
        has_processor_hook = any(
            hasattr(m, "on_processor") for m in model._modules_dict.values()
        )
        if has_processor_hook:
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=True
                )
            except Exception:
                from transformers import AutoTokenizer
                try:
                    processor = AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path, trust_remote_code=True
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Model has on_processor hooks but neither processor nor tokenizer "
                        f"found at '{pretrained_model_name_or_path}'. Save processor alongside "
                        f"the model using processor.save_pretrained(). Original error: {e}"
                    ) from e
            model.setup(processor=processor)

        return model

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args, **kwargs):
        """Migrate legacy _modules_list.{idx} keys to _modules_dict.{name}."""
        # Build index-to-name mapping from current config
        module_names = list(self.config.modules.keys())
        keys_to_rename = {}
        for key in list(state_dict.keys()):
            if key.startswith(f"{prefix}_modules_list."):
                rest = key[len(f"{prefix}_modules_list."):]
                # Extract index: "0.weight" -> idx=0, suffix="weight"
                dot_pos = rest.find(".")
                if dot_pos == -1:
                    continue
                idx_str = rest[:dot_pos]
                suffix = rest[dot_pos + 1:]
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if idx < len(module_names):
                    new_key = f"{prefix}_modules_dict.{module_names[idx]}.{suffix}"
                    keys_to_rename[key] = new_key

        for old_key, new_key in keys_to_rename.items():
            logger.info(f"Migrating checkpoint key: {old_key} -> {new_key}")
            state_dict[new_key] = state_dict.pop(old_key)


# ============================================================================
# Register with Hugging Face AutoClass for from_pretrained compatibility.
# ============================================================================

class BToks(VLM2Emb):
    """Public BToks model alias using the same runtime as VLM2Emb."""

    config_class = BToksConfig


HFAutoConfig.register("vlm2emb", VLM2EmbConfig)
HFAutoConfig.register("btoks", BToksConfig)
HFAutoModel.register(VLM2EmbConfig, VLM2Emb)
HFAutoModel.register(BToksConfig, BToks)


__all__ = [
    # Factory functions
    "create_model",
    "create_model_and_processor",
    # Helper functions
    "_infer_backbone_path",
    # Module registry (for pipeline modules)
    "AutoModule",
    # Model and config
    "BToks",
    "BToksConfig",
    "VLM2EmbConfig",
    "VLM2Emb",
]
