"""Registry system with a transformers-style Auto Classes API.

Unified component registry system compatible with HuggingFace transformers
Auto Classes style API. Registries are used by YAML configs to look up and
instantiate model modules, datasets, collators, evaluators, and trainers.

Example:
    >>> from vlm2emb import Registry
    >>>
    >>> # Create custom registry
    >>> MyRegistry = Registry("my_components")
    >>>
    >>> @MyRegistry.register("MyComponent")
    ... class MyComponent:
    ...     pass
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

from omegaconf import DictConfig

from vlm2emb.config import to_native_config
from vlm2emb.exceptions import (
    ComponentAlreadyExistsError,
    ComponentNotFoundError,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class Registry:
    """Generic component registry compatible with transformers Auto Classes style.

    Supports:
    - nn.Module (backbone, head, loss, model)
    - Dataset
    - Callable (collator, transform, factory functions)
    - Any callable object

    Attributes:
        name: Registry name (e.g., "backbone", "dataset")
        base_class: Optional base class for validation
        _components: Internal mapping of names to classes/functions
        _allow_override: Whether to allow overriding existing components

    Example:
        >>> # Create custom registry
        >>> MyRegistry = Registry("my_components")
        >>>
        >>> @MyRegistry.register("MyComponent")
        ... class MyComponent:
        ...     pass
    """

    def __init__(
        self,
        name: str,
        base_class: type | None = None,
        allow_override: bool = False,
    ):
        """Initialize an empty registry.

        Args:
            name: Registry name for identification and error messages
            base_class: Optional base class to validate registered components
            allow_override: If True, allows re-registering existing names
        """
        self.name = name
        self.base_class = base_class
        self._components: dict[str, type | Callable[..., Any]] = {}
        self._allow_override = allow_override
        # Legacy attribute aliases for backward compatibility
        self.type = name  # Old API used .type

    @overload
    def register(
        self,
        name: str | None = ...,
        component: None = ...,
        override: bool = ...,
    ) -> Callable[[_T], _T]: ...

    @overload
    def register(
        self,
        name: str | None,
        component: _T,
        override: bool = ...,
    ) -> _T: ...

    def register(
        self,
        name: str | None = None,
        component: _T | None = None,
        override: bool = False,
    ) -> _T | Callable[[_T], _T]:
        """Register a component class or factory function.

        Supports two usage patterns:
        1. Decorator: @Registry.register("name")
        2. Direct call: Registry.register("name", ComponentClass)

        Args:
            name: Component name (defaults to class/function __name__)
            component: Component class or factory function (for direct registration)
            override: Whether to allow overriding existing registration

        Returns:
            For direct call: the registered component
            For decorator: decorator function

        Raises:
            ComponentAlreadyExistsError: If component exists and override is False
            TypeError: If component doesn't inherit from base_class (when set)

        Examples:
            >>> # Decorator mode with an explicit name
            >>> @AutoModule.register("Qwen2VL")
            ... class Qwen2VLBackbone(BaseBackbone):
            ...     pass
            >>>
            >>> # Decorator mode using the class name
            >>> @AutoModule.register()
            ... class Qwen2VLBackbone(BaseBackbone):
            ...     pass
            >>>
            >>> # Direct registration
            >>> AutoModule.register("Qwen2VL", Qwen2VLBackbone)
            >>>
            >>> # Factory registration
            >>> @AutoDataset.register("mmeb_train")
            ... def load_mmeb_train(config):
            ...     return Dataset(...)
        """
        # Direct registration mode.
        if component is not None:
            component_name = name if name is not None else getattr(
                component, "__name__", type(component).__name__
            )
            self._do_register(component_name, cast(type | Callable[..., Any], component), override)
            return component

        # Decorator mode.
        def decorator(comp: _T) -> _T:
            component_name = name if name is not None else getattr(
                comp, "__name__", type(comp).__name__
            )
            self._do_register(component_name, cast(type | Callable[..., Any], comp), override)
            return comp

        return decorator

    def _do_register(
        self,
        name: str,
        component: type | Callable[..., Any],
        override: bool,
    ) -> None:
        """Validate and store one component registration.

        Args:
            name: Component name
            component: Component class or function
            override: Whether to allow overriding
        """
        allow = override or self._allow_override

        # Check for duplicate registrations unless an override is explicit.
        if name in self._components and not allow:
            raise ComponentAlreadyExistsError(
                f"Component '{name}' already registered in '{self.name}'. "
                f"Use override=True to replace it."
            )

        # Base class validation applies only to class registrations.
        if self.base_class is not None and inspect.isclass(component):
            if not issubclass(component, self.base_class):
                raise TypeError(
                    f"Component '{name}' must be a subclass of "
                    f"{self.base_class.__name__}, got {component.__name__}"
                )

        # Log overrides so accidental name collisions are visible.
        if name in self._components:
            prev_name = getattr(self._components[name], "__name__", str(self._components[name]))
            new_name = getattr(component, "__name__", str(component))
            logger.warning(
                f"Component '{name}' in '{self.name}' is being overridden. "
                f"Previous: {prev_name}, New: {new_name}"
            )

        self._components[name] = component

    def from_config(self, config: dict[str, Any] | DictConfig, **kwargs) -> Any:
        """Instantiate component from configuration dictionary.

        The config must contain a 'type' key specifying the component name.
        If the component class has a `from_config` classmethod, it will be used.
        Otherwise, other keys are passed to the component constructor.

        Args:
            config: Configuration dictionary with 'type' key
            **kwargs: Additional arguments to override config values

        Returns:
            Instantiated component

        Raises:
            ValueError: If 'type' key is missing
            ComponentNotFoundError: If component is not registered

        Example:
            >>> backbone = AutoModule.from_config({
            ...     "type": "Qwen2VL",
            ...     "model_name_or_path": "Qwen/Qwen2-VL-7B",
            ...     "device": "cuda"
            ... })
        """
        if not isinstance(config, dict) and not isinstance(config, DictConfig):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        config = to_native_config(config, resolve=True)
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        config = dict(config)

        if "type" not in config:
            raise ValueError(
                f"Config must contain 'type' key specifying component name. "
                f"Available in '{self.name}': {self.list_modules()}"
            )

        component_name = config.pop("type")
        config.update(kwargs)

        # Get the component class/function
        # Resolve the registered component class or factory.
        component = self.get(component_name)

        # If component has from_config classmethod, use it
        # Prefer a component-level from_config constructor when available.
        if hasattr(component, "from_config"):
            # Re-add type for from_config to pop if needed
            # Re-add type so component constructors can pop or inspect it.
            config_with_type = {"type": component_name, **config}
            return component.from_config(config_with_type)

        # Otherwise, call component directly with config as kwargs
        # Otherwise call the component directly with config values as kwargs.
        return component(**config)

    def build(self, name: str, *args, **kwargs) -> Any:
        """Build component instance by name.

        Args:
            name: Registered component name
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor

        Returns:
            Instantiated component

        Raises:
            ComponentNotFoundError: If component is not registered

        Example:
            >>> backbone = AutoModule.build(
            ...     "Qwen2VL",
            ...     model_name_or_path="Qwen/Qwen2-VL-7B"
            ... )
        """
        component = self.get(name)
        return component(*args, **kwargs)

    def get(self, name: str) -> type | Callable[..., Any]:
        """Get registered component class or function by name.

        Args:
            name: Component name

        Returns:
            Component class or factory function

        Raises:
            ComponentNotFoundError: If component is not registered
        """
        if name not in self._components:
            raise ComponentNotFoundError(
                f"Component '{name}' not found in '{self.name}'. "
                f"Available: {self.list_modules()}"
            )
        return self._components[name]

    def list_modules(self) -> list[str]:
        """List all registered component names.

        Returns:
            List of registered component names
        """
        return list(self._components.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a component is registered.

        Args:
            name: Component name to check

        Returns:
            True if component is registered
        """
        return name in self._components

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.is_registered(name)

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._components)

    def __repr__(self) -> str:
        """String representation."""
        return f"Registry('{self.name}', modules={self.list_modules()})"

    # ==================== Backward compatibility methods ====================

    def contains(self, name: str) -> bool:
        """Legacy alias for is_registered()."""
        return self.is_registered(name)

    def list_components(self) -> list[str]:
        """Legacy alias for list_modules()."""
        return self.list_modules()

    def register_module(
        self,
        name: str,
        module: type | Callable[..., Any],
        override: bool = False,
    ) -> None:
        """Legacy method for direct registration."""
        self._do_register(name, module, override)

    def build_from_config(self, config: dict[str, Any], **kwargs) -> Any:
        """Legacy alias for from_config()."""
        return self.from_config(config, **kwargs)


__all__ = ["Registry"]
