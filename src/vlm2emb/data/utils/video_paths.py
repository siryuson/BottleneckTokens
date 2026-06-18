"""Shared video path normalization helpers for dataset conversion and runtime."""

from __future__ import annotations


def normalize_youcook2_video_path(value: object) -> str:
    """Normalize a YouCook2 source video path to the raw tar member key."""

    normalized = str(value).strip().replace("\\", "/")
    normalized = normalized.split("//")[-1]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


__all__ = ["normalize_youcook2_video_path"]
