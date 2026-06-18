"""Shared runtime transform helpers for migrated datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vlm2emb.data.datasets.const import decode_image, extract_image_bytes

TRAIN_SAMPLE_SIDES = ("query", "positive", "negative")
INSTRUCTION_BODY_SEPARATORS = {"none", "space", "newline"}


def normalize_side_transform_mapping(
    values: Mapping[str, Any] | None,
    *,
    dataset_name: str,
    allowed_keys: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Validate a side-scoped default-transform config mapping."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{dataset_name} transform config must be a side-scoped mapping.")

    unknown_sides = sorted(set(values) - set(TRAIN_SAMPLE_SIDES))
    if unknown_sides:
        raise ValueError(f"Unsupported {dataset_name} transform sides: {unknown_sides}")

    normalized: dict[str, dict[str, Any]] = {}
    for side, side_values in values.items():
        if side_values is None:
            normalized[side] = {}
            continue
        if not isinstance(side_values, Mapping):
            raise TypeError(f"{dataset_name} transform.{side} must be a mapping.")
        allowed = allowed_keys.get(side, set())
        unknown_keys = sorted(set(side_values) - allowed)
        if unknown_keys:
            raise ValueError(f"Unsupported {dataset_name} transform.{side} keys: {unknown_keys}")
        normalized[side] = dict(side_values)
    return normalized


def normalize_instruction_text(
    value: Any,
    *,
    dataset_name: str,
    name: str,
    default: str = "",
    allow_empty: bool = True,
) -> str:
    """Validate one configurable instruction string."""

    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{dataset_name} {name} must be a string.")
    if not value.strip() and not allow_empty:
        raise ValueError(f"{dataset_name} {name} must be a non-empty string.")
    return value


def normalize_instruction_body_separator(
    value: Any,
    *,
    dataset_name: str,
    name: str = "instruction_body_separator",
    default: str = "none",
) -> str:
    """Validate how a side instruction is joined with the side body text."""

    if value is None:
        value = default
    if not isinstance(value, str) or value not in INSTRUCTION_BODY_SEPARATORS:
        raise ValueError(
            f"{dataset_name} {name} must be one of {sorted(INSTRUCTION_BODY_SEPARATORS)}."
        )
    return str(value)


def join_instruction_body(instruction: str, body: str, *, separator: str) -> str:
    """Join an instruction and body without changing either string's wording."""

    if not instruction:
        return body
    if not body:
        return instruction
    if separator == "none":
        return f"{instruction}{body}"
    if separator == "space":
        return f"{instruction} {body}"
    if separator == "newline":
        return f"{instruction}\n{body}"
    raise ValueError(f"Unsupported instruction_body_separator value: {separator}")


def image_path_from_record(
    record: Mapping[str, Any],
    *,
    image_field: str = "image",
    fallback_path_fields: tuple[str, ...] = (),
) -> str | None:
    """Resolve an image path from a raw record and optional fallback fields."""

    image = record.get(image_field)
    if isinstance(image, Mapping):
        path = image.get("path")
        if path:
            return str(path)
    for field in fallback_path_fields:
        path = record.get(field)
        if path:
            return str(path)
    return None


def image_media_from_record(
    record: Mapping[str, Any],
    *,
    image_field: str = "image",
    fallback_path_fields: tuple[str, ...] = (),
    missing_message: str = "Row is missing required image bytes",
) -> list[dict[str, Any]]:
    """Build one image media slot from a raw record."""

    image_bytes = extract_image_bytes(record.get(image_field, {}))
    if image_bytes is None:
        raise ValueError(missing_message)
    metadata: dict[str, Any] = {}
    path = image_path_from_record(
        record,
        image_field=image_field,
        fallback_path_fields=fallback_path_fields,
    )
    if path is not None:
        metadata["path"] = path
    return [{"kind": "image", "content": decode_image(image_bytes), "metadata": metadata}]


def ensure_visual_token(
    text: str | None,
    *,
    token: str,
    legacy_tokens: tuple[str, ...] = (),
    separator: str = "\n",
    position: str = "prepend",
) -> str:
    """Ensure one visual token is present when runtime input includes media."""

    normalized = text or ""
    for legacy_token in legacy_tokens:
        normalized = normalized.replace(legacy_token, token)
    if token in normalized:
        return normalized
    if not normalized:
        return token
    if position == "append":
        return f"{normalized}{separator}{token}"
    return f"{token}{separator}{normalized}"


def transform_eval_sample_with_media_token(
    sample: dict[str, Any],
    *,
    token: str,
    text_prefix: str = "",
    legacy_tokens: tuple[str, ...] = (),
    separator: str = "\n",
    token_position: str = "prepend",
) -> dict[str, Any]:
    """Inject the appropriate modality token into retrieval eval samples."""

    transformed = dict(sample)
    images = transformed.get("images", [])
    if not images:
        return transformed

    text = transformed.get("text", "") or ""
    if text_prefix and not text.startswith(text_prefix):
        text = f"{text_prefix}{text}"
    transformed["text"] = ensure_visual_token(
        text,
        token=token,
        legacy_tokens=legacy_tokens,
        separator=separator,
        position=token_position,
    )
    return transformed


def transform_eval_sample_with_text_prefix(
    sample: dict[str, Any],
    *,
    text_prefix: str,
) -> dict[str, Any]:
    """Prefix retrieval text without introducing a local closure."""

    transformed = dict(sample)
    text = transformed.get("text", "") or ""
    if text_prefix and not text.lstrip().startswith(text_prefix):
        transformed["text"] = f"{text_prefix}{text}"
    else:
        transformed["text"] = text
    return transformed


__all__ = [
    "INSTRUCTION_BODY_SEPARATORS",
    "TRAIN_SAMPLE_SIDES",
    "ensure_visual_token",
    "image_media_from_record",
    "image_path_from_record",
    "join_instruction_body",
    "normalize_instruction_body_separator",
    "normalize_instruction_text",
    "normalize_side_transform_mapping",
    "transform_eval_sample_with_media_token",
    "transform_eval_sample_with_text_prefix",
]
