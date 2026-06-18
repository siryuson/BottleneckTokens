"""Dataset schema constants for the BToks public runtime.

This module defines constants and small runtime validation helpers used by
dataset implementations.

Architecture:
============================================
With Lance datasets, images are decoded in Dataset.__getitem__() and returned
as PIL.Image objects directly. The Collator no longer handles image loading.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from PIL import Image

# ============================================================================
# Image token constants
# ============================================================================

STANDARD_IMAGE_TOKEN = "<|image_pad|>"  # Standard format (Qwen)
PHI3V_IMAGE_TOKEN = "<|image_1|>"      # Original format in MMEB
STANDARD_VIDEO_TOKEN = "<|video_pad|>"  # Standard video format (Qwen)
LEGACY_IMAGE_TOKENS = (PHI3V_IMAGE_TOKEN, "<image>")
LEGACY_VIDEO_TOKENS = ("<video>",)
STANDARD_VISUAL_TOKENS = (
    STANDARD_IMAGE_TOKEN,
    STANDARD_VIDEO_TOKEN,
)
LANCE_MAX_BYTES_PER_FILE = 2 * 1024 * 1024 * 1024

def has_visual_token(text: str | None) -> bool:
    """Return whether text contains a final standardized visual token."""
    normalized = text or ""
    return any(token in normalized for token in STANDARD_VISUAL_TOKENS)


def validate_encoding_sample(
    sample: dict[str, Any],
    *,
    sample_name: str = "sample",
    allow_visual_token_without_media: bool = False,
) -> dict[str, Any]:
    """Validate one model-facing encoding sample for token/media alignment."""
    text = sample.get("text", "")
    images = sample.get("images", [])
    media_metadata = sample.get("media_metadata")

    if not isinstance(text, str):
        raise TypeError(f"{sample_name}.text must be str, got {type(text).__name__}")
    if not isinstance(images, list):
        raise TypeError(f"{sample_name}.images must be list, got {type(images).__name__}")
    if media_metadata is not None and not isinstance(media_metadata, dict):
        raise TypeError(
            f"{sample_name}.media_metadata must be dict when present, got {type(media_metadata).__name__}"
        )

    has_media = len(images) > 0
    has_visual = has_visual_token(text)
    if has_media and not has_visual:
        raise ValueError(f"{sample_name} has media but text contains no visual token")
    if has_visual and not has_media and not allow_visual_token_without_media:
        raise ValueError(f"{sample_name} has visual token(s) in text but no media")
    return sample


def validate_runtime_sample(
    sample: dict[str, Any],
    *,
    sample_name: str = "sample",
    allow_visual_token_without_media: bool = False,
) -> dict[str, Any]:
    """Validate a runtime dataset sample after all runtime transforms."""
    if "text" in sample and "images" in sample:
        return validate_encoding_sample(
            sample,
            sample_name=sample_name,
            allow_visual_token_without_media=allow_visual_token_without_media,
        )

    for field in ("query", "candidate", "positive", "negative"):
        if field in sample and isinstance(sample[field], dict):
            multimodal = sample[field]
            text = multimodal.get("text", "")
            media = multimodal.get("media")
            if media is None:
                media = multimodal.get("images", [])
            if not isinstance(text, str):
                raise TypeError(
                    f"{sample_name}.{field}.text must be str, got {type(text).__name__}"
                )
            if not isinstance(media, list):
                raise TypeError(
                    f"{sample_name}.{field}.media must be list, got {type(media).__name__}"
                )
            has_media = len(media) > 0
            has_visual = has_visual_token(text)
            if has_media and not has_visual:
                raise ValueError(f"{sample_name}.{field} has media but text contains no visual token")
            if has_visual and not has_media and not allow_visual_token_without_media:
                raise ValueError(f"{sample_name}.{field} has visual token(s) in text but no media")
    return sample


# ============================================================================
# Image utility functions
# ============================================================================


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes to PIL.Image in RGB mode."""
    return Image.open(BytesIO(data)).convert("RGB")


def extract_image_bytes(image_data: Any) -> bytes | None:
    """Extract image bytes from various formats (dict struct, raw bytes)."""
    if isinstance(image_data, dict):
        return image_data.get("bytes")
    elif isinstance(image_data, bytes):
        return image_data
    return None


def normalize_text_whitespace(
    text: str | None,
    *,
    ensure_trailing_newline: bool = False,
) -> str:
    """Normalize text whitespace without changing textual semantics."""
    if text is None:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))
    normalized = normalized.lstrip(" \t").rstrip(" \t")
    if not normalized.strip(" \t\n"):
        return ""
    if ensure_trailing_newline:
        return normalized.rstrip("\n") + "\n"
    return normalized


def canonicalize_multimodal_text(
    text: str | None,
    *,
    token: str | None = None,
    legacy_tokens: tuple[str, ...] = (),
    ensure_trailing_newline: bool = False,
) -> str:
    """Standardize tokens and normalize standalone token layout."""
    normalized = text or ""
    if token:
        for legacy_token in legacy_tokens:
            normalized = normalized.replace(legacy_token, token)
        normalized = _move_standalone_token_to_front(normalized, token)
    return normalize_text_whitespace(
        normalized,
        ensure_trailing_newline=ensure_trailing_newline,
    )


def _move_standalone_token_to_front(text: str, token: str) -> str:
    """Move standalone modality token to the first line when safe."""
    normalized = normalize_text_whitespace(text, ensure_trailing_newline=False)
    trailing_newlines = len(normalized) - len(normalized.rstrip("\n"))
    normalized_core = normalized.rstrip("\n")
    if not normalized_core or token not in normalized_core or normalized_core.count(token) != 1:
        return normalized

    trailing_suffix = "\n" * trailing_newlines

    # Case 1: token already occupies a dedicated line somewhere in the text.
    token_line_pattern = re.compile(rf"(?m)^[ \t]*{re.escape(token)}[ \t]*$")
    if token_line_pattern.search(normalized_core):
        lines = [line.strip(" \t") for line in normalized_core.split("\n")]
        body = "\n".join(line for line in lines if line != token).strip(" \t\n")
        result = f"{token}\n{body}" if body else token
        return result + trailing_suffix

    # Case 2: token is a leading marker on the same line as body text.
    leading_pattern = re.compile(rf"^[ \t]*{re.escape(token)}[ \t]+(?P<body>.+)$", re.DOTALL)
    match = leading_pattern.match(normalized_core)
    if match:
        body = match.group("body").strip(" \t\n")
        result = f"{token}\n{body}" if body else token
        return result + trailing_suffix

    # Case 3: token is appended at the end of a line or string.
    line_end_pattern = re.compile(
        rf"^(?P<prefix>[^\n]*?)[ \t]+{re.escape(token)}[ \t]*(?:\n(?P<rest>.*))?$",
        re.DOTALL,
    )
    match = line_end_pattern.match(normalized_core)
    if match:
        parts = [
            (match.group("prefix") or "").strip(" \t\n"),
            (match.group("rest") or "").strip(" \t\n"),
        ]
        body = "\n".join(part for part in parts if part)
        result = f"{token}\n{body}" if body else token
        return result + trailing_suffix

    return normalized
