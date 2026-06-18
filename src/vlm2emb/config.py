"""OmegaConf-based configuration loader for VLM2Emb/BToks.

This module provides configuration loading with:
- YAML configuration file support
- Configuration inheritance via the ``_inherit_`` directive
- Circular dependency detection for inherited configs
- Type-safe DictConfig/ListConfig return values
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from omegaconf.errors import InterpolationKeyError
from yaml import YAMLError

from vlm2emb.exceptions import (
    ConfigError,
    ConfigInheritanceError,
    ConfigInterpolationError,
    ConfigNotFoundError,
    ConfigSyntaxError,
)

if TYPE_CHECKING:
    pass

# Type alias for OmegaConf config containers.
Config: TypeAlias = DictConfig | ListConfig

logger = logging.getLogger(__name__)


def resolve_interpolations(
    config: Config,
    config_path: str | Path | None = None,
) -> Config:
    """Resolve all interpolations in configuration.

    This function resolves all ${param} references in the configuration
    using OmegaConf's built-in interpolation mechanism.

    Args:
        config: Loaded configuration DictConfig with interpolations
        config_path: Path to config file (for error messages)

    Returns:
        DictConfig with all interpolations resolved

    Raises:
        ConfigInterpolationError: If interpolation cannot be resolved
        InterpolationKeyError: If referenced parameter doesn't exist

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({
        ...     "experiment": {"name": "test"},
        ...     "output_dir": "experiments/${experiment.name}"
        ... })
        >>> resolved = resolve_interpolations(config)
        >>> print(resolved.output_dir)
        experiments/test
    """
    try:
        # OmegaConf.resolve() modifies config in-place.
        OmegaConf.resolve(config)
        return config
    except InterpolationKeyError as e:
        # Format error message with config-path context.
        error_msg = "Failed to resolve interpolation in configuration"

        # Try to extract the unresolved reference from the error message.
        error_str = str(e)
        unresolved_ref = None
        if "Interpolation key" in error_str and "not found" in error_str:
            # Format: "Interpolation key 'foo.bar' not found"
            match = re.search(r"Interpolation key '([^']+)'", error_str)
            if match:
                unresolved_ref = match.group(1)

        raise ConfigInterpolationError(
            message=error_msg,
            file_path=config_path,
            unresolved_reference=unresolved_ref,
            details={
                "original_error": str(e),
                "original_error_type": type(e).__name__,
            },
        ) from e


def _normalize_path(path: str | Path) -> Path:
    """Normalize a path to absolute resolved form.

    Args:
        path: The path to normalize.

    Returns:
        Resolved absolute Path object.
    """
    return Path(path).resolve()


def _resolve_inherit_path(config_dir: Path, inherit_value: str) -> Path:
    """Resolve _inherit_ value relative to config file's directory.

    Args:
        config_dir: Directory containing the child config file.
        inherit_value: The _inherit_ value (relative or absolute path).

    Returns:
        Resolved absolute path to the base config file.
    """
    inherit_path = Path(inherit_value)
    if inherit_path.is_absolute():
        return inherit_path.resolve()
    # Resolve relative paths against the child config file's directory.
    return (config_dir / inherit_path).resolve()


class ConfigLoader:
    """Configuration loader with OmegaConf integration.

    Provides YAML configuration loading with inheritance support.

    Example:
        >>> loader = ConfigLoader("configs/base.yaml")
        >>> config = loader.load_with_inheritance("configs/experiment.yaml")
        >>> print(config.model.backbone)
    """

    def __init__(
        self,
        base_config_path: str | Path | None = None,
    ) -> None:
        """Initialize the configuration loader.

        Args:
            base_config_path: Optional path to base configuration file.
                If provided, this config will be loaded and used as the
                base for all subsequent loads.
        """
        self._base_config: Config | None = None

        if base_config_path is not None:
            resolved = _normalize_path(base_config_path)
            self._base_config = self._load_single_config(str(resolved))

    def _load_single_config(self, config_path: str) -> Config:
        """Load a single YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            OmegaConf DictConfig object.

        Raises:
            ConfigNotFoundError: If the file does not exist.
            ConfigSyntaxError: If the YAML syntax is invalid.
            ConfigError: For other configuration-related errors.
        """
        path = Path(config_path)
        resolved_path = _normalize_path(path)

        # Check if file exists before calling OmegaConf for clearer errors.
        if not resolved_path.exists():
            raise ConfigNotFoundError(attempted_path=str(resolved_path))

        try:
            config = OmegaConf.load(resolved_path)
            logger.debug(f"Successfully loaded config: {config_path}")
            # Ensure callers always receive a config container, even for scalar roots.
            if isinstance(config, DictConfig):
                return config
            return OmegaConf.create({"_root_": config})
        except YAMLError as e:
            # Only catch YAML-specific parsing errors here.
            raise ConfigSyntaxError(
                file_path=str(resolved_path),
                original_error=e,
                suggestion="Check YAML syntax and indentation",
            ) from e
        except PermissionError as e:
            raise ConfigError(
                message=f"Permission denied accessing config file: {resolved_path}",
                file_path=str(resolved_path),
                details={"error": str(e)},
            ) from e
        except OSError as e:
            raise ConfigError(
                message=f"I/O error reading config file: {resolved_path}",
                file_path=str(resolved_path),
                details={"error": str(e)},
            ) from e

    def load_with_inheritance(
        self,
        config_path: str,
        resolve_interpolation: bool = True,
    ) -> Config:
        """Load configuration with inheritance support.

        Supports `_inherit_` directive for configuration inheritance at ANY level.
        The directive specifies the path to the base configuration file,
        which can be relative (resolved relative to the config file's directory)
        or absolute.

        Example:
            # configs/experiment.yaml
            _inherit_: ../base.yaml  # relative to config's directory
            model:
                backbone: qwen2vl_7b

            # Nested inheritance at any level:
            training:
              optimizer:
                _inherit_: configs/optimizer/adamw.yaml
                lr: 0.0001

        Args:
            config_path: Path to the configuration file.
            resolve_interpolation: Whether to resolve variable interpolations.
                Defaults to True.

        Returns:
            Merged DictConfig with inheritance applied.

        Raises:
            ConfigNotFoundError: If a configuration file is not found.
            ConfigSyntaxError: If YAML syntax is invalid.
            ConfigInheritanceError: If circular inheritance is detected.
        """
        original_config_path = _normalize_path(config_path)
        current_config = self._load_config_tree(
            original_config_path,
            active_stack=[],
            active_stack_set=set(),
        )

        # Apply process-level base config when one was provided to the loader.
        if self._base_config is not None:
            current_config = OmegaConf.merge(self._base_config, current_config)

        # Resolve variable interpolations only after inheritance has been applied.
        if resolve_interpolation:
            current_config = resolve_interpolations(
                current_config, config_path=str(original_config_path)
            )

        return current_config

    def _load_config_tree(
        self,
        config_path: Path,
        active_stack: list[Path],
        active_stack_set: set[Path],
    ) -> Config:
        """Load one config file and expand all inheritance under it."""
        normalized_path = _normalize_path(config_path)
        self._ensure_not_in_active_chain(
            normalized_path,
            active_stack,
            active_stack_set,
        )

        active_stack.append(normalized_path)
        active_stack_set.add(normalized_path)
        try:
            current_config = self._load_single_config(str(normalized_path))
            return self._process_nested_inheritance(
                current_config,
                normalized_path.parent,
                active_stack,
                active_stack_set,
            )
        finally:
            active_stack.pop()
            active_stack_set.remove(normalized_path)

    def _process_nested_inheritance(
        self,
        config: Config,
        config_dir: Path,
        active_stack: list[Path],
        active_stack_set: set[Path],
    ) -> Config:
        """Process inheritance recursively at any nested level.

        Args:
            config: The configuration DictConfig to process.
            config_dir: Directory for resolving relative inheritance paths.

        Returns:
            DictConfig with all inheritance processed.

        Raises:
            ConfigInheritanceError: If circular inheritance is detected.
            ConfigNotFoundError: If inherited config file doesn't exist.
        """
        result = OmegaConf.create(config)

        if OmegaConf.is_dict(result):
            if self._has_inheritance(result):
                merged = self._merge_inherited_section(
                    result,
                    config_dir,
                    active_stack,
                    active_stack_set,
                )
                return self._process_nested_inheritance(
                    merged,
                    config_dir,
                    active_stack,
                    active_stack_set,
                )

            for key in list(result.keys()):
                if OmegaConf.is_missing(result, key):
                    continue
                if OmegaConf.is_interpolation(result, key):
                    continue

                value = result[key]
                if OmegaConf.is_dict(value) or OmegaConf.is_list(value):
                    result[key] = self._process_nested_inheritance(
                        value,
                        config_dir,
                        active_stack,
                        active_stack_set,
                    )

        elif OmegaConf.is_list(result):
            for index, item in enumerate(result):
                if OmegaConf.is_dict(item) or OmegaConf.is_list(item):
                    result[index] = self._process_nested_inheritance(
                        item,
                        config_dir,
                        active_stack,
                        active_stack_set,
                    )

        return result

    def _merge_inherited_section(
        self,
        config: Config,
        config_dir: Path,
        active_stack: list[Path],
        active_stack_set: set[Path],
    ) -> Config:
        """Merge a config section with its inherited base.

        Args:
            config: Config with _inherit_ directive.
            config_dir: Directory for resolving relative paths.

        Returns:
            Merged config with _inherit_ removed.

        Raises:
            ConfigInheritanceError: If circular inheritance detected.
            ConfigNotFoundError: If base config not found.
        """
        if not self._has_inheritance(config):
            return config

        inherit_value = self._get_inheritance_path(config)
        if inherit_value is None:
            return config

        # Resolve the inherited config path relative to the current config.
        base_path = _resolve_inherit_path(config_dir, inherit_value)
        base_config = self._load_config_tree(
            base_path,
            active_stack,
            active_stack_set,
        )

        # Create a copy and remove the _inherit_ key before merging.
        config_without_inherit = OmegaConf.create(config)
        with open_dict(config_without_inherit):
            if "_inherit_" in config_without_inherit:
                del config_without_inherit["_inherit_"]

        # Merge base first, then override with the current section.
        merged = OmegaConf.merge(base_config, config_without_inherit)

        # Ensure _inherit_ is removed from the merged result in case base had it.
        if OmegaConf.is_dict(merged) and "_inherit_" in merged:
            with open_dict(merged):
                del merged["_inherit_"]

        return merged

    def _ensure_not_in_active_chain(
        self,
        config_path: Path,
        active_stack: list[Path],
        active_stack_set: set[Path],
    ) -> None:
        """Raise a circular inheritance error for the active recursion chain."""
        if config_path not in active_stack_set:
            return

        cycle_start = active_stack.index(config_path)
        cycle_paths = [str(path) for path in active_stack[cycle_start:]] + [str(config_path)]
        raise ConfigInheritanceError(
            message="Circular inheritance detected",
            inheritance_chain=cycle_paths,
            error_type="circular",
        )

    def _has_inheritance(self, config: Config) -> bool:
        """Check if config has inheritance directive.

        Args:
            config: DictConfig to check.

        Returns:
            True if _inherit_ key exists (as plain key or interpolation).
        """
        # Check if _inherit_ is present as a key in the config.
        if isinstance(config, DictConfig) and "_inherit_" in config:
            # Treat None and MISSING as absent inheritance directives.
            try:
                value = config._get_node("_inherit_")
                if value is not None and not OmegaConf.is_missing(config, "_inherit_"):
                    return True
            except Exception:
                pass
        return False

    def _get_inheritance_path(self, config: Config) -> str | None:
        """Get the inheritance path from config.

        Args:
            config: DictConfig to check.

        Returns:
            The inheritance path string, or None if not set.
        """
        if not isinstance(config, DictConfig):
            return None
        if not self._has_inheritance(config):
            return None
        # Read the raw value without resolving interpolations.
        inherit_node = config._get_node("_inherit_")
        if inherit_node is not None and hasattr(inherit_node, "value"):
            return inherit_node.value
        return OmegaConf.select(config, "_inherit_")

    def load_preset(
        self,
        preset_name: str,
        custom_overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Load a preset configuration from configs/presets/.

        Args:
            preset_name: Name of the preset (without .yaml extension).
            custom_overrides: Optional dictionary of overrides to apply.

        Returns:
            DictConfig for the preset with overrides applied.
        """
        preset_path = f"configs/presets/{preset_name}.yaml"
        config = self.load_with_inheritance(preset_path)

        if custom_overrides:
            override_config = OmegaConf.create(custom_overrides)
            config = OmegaConf.merge(config, override_config)

        return config

    def load_modular(
        self,
        components: dict[str, str],
        training: str = "default",
    ) -> Config:
        """Load configuration from modular component files.

        Args:
            components: Dict mapping component types to names.
                Example: {"backbone": "qwen2vl_7b", "head": "embedding_head"}
            training: Training configuration name. Set to None to skip.

        Returns:
            Merged DictConfig from all component configs.

        Raises:
            ConfigError: If no components and no training config are specified.
        """
        configs: list[Config] = []

        for component_type, component_name in components.items():
            config_path = f"configs/{component_type}/{component_name}.yaml"
            cfg = self._load_single_config(config_path)
            configs.append(cfg)

        if training is not None:
            training_path = f"configs/training/{training}.yaml"
            cfg = self._load_single_config(training_path)
            configs.append(cfg)

        # Validate that at least one config source was selected.
        if not configs:
            raise ConfigError(
                message="No configuration files to merge. "
                        "At least one component must be specified in 'components' "
                        "or 'training' must be provided.",
                details={
                    "components_provided": bool(components),
                    "training_provided": training is not None,
                },
            )

        # Merge all configs sequentially so later files override earlier ones.
        result: Config = configs[0]
        for cfg in configs[1:]:
            result = OmegaConf.merge(result, cfg)

        return result


def load_config(config_path: str | Path) -> Config:
    """Load a single configuration file with inheritance support.

    This convenience function loads a configuration file and processes any
    `_inherit_` directives it contains at ANY level. The inheritance paths are
    resolved relative to the configuration file's directory.

    This is the unified API for configuration loading - inheritance is enabled
    by default and interpolations are resolved.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        DictConfig with inheritance applied and interpolations resolved.

    Raises:
        ConfigNotFoundError: If the file does not exist.
        ConfigSyntaxError: If the YAML syntax is invalid.
        ConfigInheritanceError: If circular inheritance is detected.
    """
    loader = ConfigLoader()
    return loader.load_with_inheritance(str(config_path), resolve_interpolation=True)


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply CLI overrides to configuration.

    Supports full OmegaConf dotlist syntax:

    - `key=value` - set a value
    - `key=null` - set a value to null
    - `+key=value` - add a new key
    - `~key` - delete a key
    - `key.nested=value` - set a nested key

    Args:
        config: Loaded configuration DictConfig
        overrides: List of override strings in dotlist format

    Returns:
        Config with overrides applied

    Example:
        >>> config = load_config("config.yaml")
        >>> config = apply_overrides(config, [
        ...     "training.learning_rate=1e-4",
        ...     "model.backbone.type=Qwen2VL",
        ...     "+experiment.tag=test",
        ... ])
    """
    if not overrides:
        return config

    override_config = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(config, override_config)


def to_native_config(
    config: Config | dict[str, Any] | list[Any],
    *,
    resolve: bool = True,
) -> dict[str, Any] | list[Any]:
    """Convert OmegaConf containers to native Python containers."""
    if isinstance(config, (dict, list)):
        return config

    if isinstance(config, (DictConfig, ListConfig)):
        normalized_config = OmegaConf.create(config)
        if resolve:
            OmegaConf.resolve(normalized_config)
        native_config = OmegaConf.to_container(
            normalized_config,
            resolve=resolve,
        )
        if isinstance(native_config, (dict, list)):
            return native_config

    raise TypeError(
        "to_native_config() expects a DictConfig, ListConfig, dict, or list"
    )


__all__ = [
    # Core API
    "load_config",
    "apply_overrides",
    "to_native_config",
    # Advanced
    "ConfigLoader",
    "resolve_interpolations",
    # Types (re-exported from OmegaConf)
    "Config",
    "DictConfig",
    "ListConfig",
    # Exceptions
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigSyntaxError",
    "ConfigInheritanceError",
    "ConfigInterpolationError",
]
