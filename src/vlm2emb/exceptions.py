"""Custom exceptions for the BToks public runtime.

This module defines all framework-specific exceptions including:
- Base exception classes
- Registry and component errors
- Configuration errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# =============================================================================
# Base exception classes.
# =============================================================================

class VLM2EmbError(Exception):
    """Base exception for all public runtime errors."""

    pass


# =============================================================================
# Registry and component errors.
# =============================================================================

class RegistryError(VLM2EmbError):
    """Base class for registry-related errors."""

    pass


class ComponentNotFoundError(RegistryError):
    """Raised when a component name is not registered in the requested registry."""

    pass


class ComponentAlreadyExistsError(RegistryError):
    """Raised when registering a duplicate component name without override."""

    pass


class ComponentError(VLM2EmbError):
    """Base class for component initialization or execution errors."""

    pass


class InterfaceValidationError(ComponentError):
    """Raised when a component does not satisfy the required interface."""

    pass


class MissingMethodError(InterfaceValidationError):
    """Raised when a component is missing one or more required methods."""

    def __init__(self, component_name: str, missing_methods: set[str]) -> None:
        """Initialize an error that lists missing interface methods.

        Args:
            component_name: Name of the component that failed validation.
            missing_methods: Set of method names that are required but absent.
        """
        self.component_name = component_name
        self.missing_methods = missing_methods
        super().__init__(
            f"Component '{component_name}' is missing required methods: "
            f"{', '.join(sorted(missing_methods))}"
        )


class InvalidSignatureError(InterfaceValidationError):
    """Raised when a component method has an invalid signature."""

    pass


class InputError(VLM2EmbError):
    """Raised for invalid user or runtime inputs."""

    pass


class ModelError(VLM2EmbError):
    """Raised for model construction, loading, or execution errors."""

    pass


# =============================================================================
# Configuration errors.
# =============================================================================

class ConfigError(Exception):
    """Configuration loading or parsing error.

    Raised when:
    - YAML syntax is invalid.
    - Configuration structure is malformed.
    - Required fields are missing.

    Attributes:
        file_path: Path to the configuration file that caused the error.
        line_number: Line number where the error occurred, if available.
    """

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        line_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a configuration error with optional source location.

        Args:
            message: Human-readable error message.
            file_path: Path to the config file, when known.
            line_number: Line number where the error occurred, when known.
            details: Structured diagnostic details for callers.
        """
        self.file_path = str(file_path) if file_path else None
        self.line_number = line_number
        self.details = details or {}

        # Include file and line context in the exception string when available.
        if file_path and line_number:
            formatted_message = f"{message} (File: {file_path}, Line: {line_number})"
        elif file_path:
            formatted_message = f"{message} (File: {file_path})"
        else:
            formatted_message = message

        super().__init__(formatted_message)


class ConfigNotFoundError(ConfigError):
    """Configuration file was not found.

    Raised when the specified configuration file does not exist.

    Attributes:
        attempted_path: The path that was attempted but not found.
    """

    def __init__(
        self,
        attempted_path: str | Path,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an error for a missing configuration file.

        Args:
            attempted_path: Path that was attempted.
            details: Additional structured error details.
        """
        self.attempted_path = str(attempted_path)
        message = f"Configuration file not found: {self.attempted_path}"

        super().__init__(
            message=message,
            file_path=attempted_path,
            details=details or {"attempted_path": self.attempted_path},
        )


class ConfigSyntaxError(ConfigError):
    """Raised when a configuration file has invalid YAML syntax.

    Attributes:
        original_error: The original YAML parsing error.
        suggestion: Suggested fix for the error.
    """

    def __init__(
        self,
        file_path: str | Path,
        original_error: Exception,
        suggestion: str | None = None,
    ) -> None:
        """Initialize an error for invalid YAML syntax.

        Args:
            file_path: Path to the config file.
            original_error: The original YAML parsing error.
            suggestion: Optional suggested fix.
        """
        self.original_error = original_error
        self.suggestion = suggestion

        # Try to extract line number from YAML parser diagnostics.
        line_number: int | None = None
        error_str = str(original_error)
        if "line" in error_str.lower():
            import re

            match = re.search(r"line (\d+)", error_str.lower())
            if match:
                line_number = int(match.group(1))

        message = f"Syntax error in configuration file: {file_path}"
        if suggestion:
            message = f"{message}\n  Suggestion: {suggestion}"

        details = {
            "original_error_type": type(original_error).__name__,
            "original_error_message": str(original_error),
        }
        if suggestion:
            details["suggestion"] = suggestion

        super().__init__(
            message=message,
            file_path=file_path,
            line_number=line_number,
            details=details,
        )


class ConfigInheritanceError(ConfigError):
    """Raised when there is an error in configuration inheritance.

    This includes:
    - Circular inheritance dependencies.
    - Invalid inheritance paths.
    - Missing base configurations.

    Attributes:
        inheritance_chain: The chain of inheritance that caused the error.
        error_type: Type of inheritance error: circular, invalid, or missing.
    """

    def __init__(
        self,
        message: str,
        inheritance_chain: list[str] | None = None,
        error_type: str = "invalid",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an error for configuration inheritance failures.

        Args:
            message: Human-readable error message.
            inheritance_chain: Chain of inherited files involved in the error.
            error_type: Error category such as circular, invalid, or missing.
            details: Additional structured error details.
        """
        self.inheritance_chain = inheritance_chain or []
        self.error_type = error_type

        full_message = message
        if inheritance_chain:
            # Include the inheritance chain in the message for easier debugging.
            chain_str = " -> ".join(self.inheritance_chain)
            full_message = f"{message}\n  Inheritance chain: {chain_str}"

        super().__init__(
            message=full_message,
            details=details or {
                "inheritance_chain": self.inheritance_chain,
                "error_type": error_type,
            },
        )


class ConfigValidationError(ConfigError):
    """Configuration validation failed.

    Raised when configuration values don't meet validation requirements.
    """

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an error for invalid configuration values.

        Args:
            message: Human-readable error message.
            file_path: Path to the config file, when known.
            field_path: Dot-path to the field that failed validation.
            details: Additional structured error details.
        """
        self.field_path = field_path
        self.details = details or {}

        if field_path:
            formatted_message = f"{message} (Field: {field_path})"
            if file_path:
                formatted_message = f"{message} (Field: {field_path}, File: {file_path})"
        else:
            formatted_message = message

        super().__init__(
            message=formatted_message,
            file_path=file_path,
            details=self.details,
        )


class ConfigInterpolationError(ConfigError):
    """Configuration interpolation resolution failed.

    Raised when OmegaConf cannot resolve variable interpolations in configuration.

    Attributes:
        unresolved_reference: The reference that could not be resolved.
    """

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        unresolved_reference: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an error for unresolved OmegaConf interpolation.

        Args:
            message: Human-readable error message.
            file_path: Path to the config file, when known.
            unresolved_reference: The interpolation reference that failed to resolve.
            details: Additional structured error details.
        """
        self.unresolved_reference = unresolved_reference
        self.details = details or {}

        if unresolved_reference:
            formatted_message = f"{message} (Unresolved: {unresolved_reference})"
            if file_path:
                formatted_message = f"{message} (Unresolved: {unresolved_reference}, File: {file_path})"
        else:
            formatted_message = message

        details_with_ref = dict(self.details)
        if unresolved_reference:
            details_with_ref["unresolved_reference"] = unresolved_reference

        super().__init__(
            message=formatted_message,
            file_path=file_path,
            details=details_with_ref,
        )


__all__ = [
    # Base exceptions
    "VLM2EmbError",
    # Registry and component
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentAlreadyExistsError",
    "ComponentError",
    "InterfaceValidationError",
    "MissingMethodError",
    "InvalidSignatureError",
    "InputError",
    "ModelError",
    # Config exceptions
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigSyntaxError",
    "ConfigInheritanceError",
    "ConfigValidationError",
    "ConfigInterpolationError",
]
