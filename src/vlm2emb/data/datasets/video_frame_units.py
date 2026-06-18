"""Runtime helpers for pre-extracted video frame-units."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

import numpy as np
from PIL import Image


def decode_frame_bytes(frame: bytes) -> Image.Image:
    """Decode one stored frame image."""

    return Image.open(BytesIO(frame)).convert("RGB")


def sample_preextracted_frame_unit(
    record: Mapping[str, Any],
    *,
    num_frames: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Read and sample frames from a joined frame-unit record."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive for pre-extracted video frames.")
    frames = record.get("frames")
    if not isinstance(frames, list):
        raise TypeError("Frame-unit row is missing required frames list.")
    if not frames:
        raise ValueError("Frame-unit row resolved to zero stored frames.")

    if len(frames) > num_frames:
        relative_indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int).tolist()
    else:
        relative_indices = list(range(len(frames)))
    images = [decode_frame_bytes(frames[index]) for index in relative_indices]

    stored_sampled_indices = record.get("sampled_indices")
    if isinstance(stored_sampled_indices, list) and stored_sampled_indices:
        selected_source_indices = [
            int(stored_sampled_indices[index])
            for index in relative_indices
            if index < len(stored_sampled_indices)
        ]
    else:
        selected_source_indices = relative_indices

    stored_timestamps = record.get("sampled_timestamps")
    if isinstance(stored_timestamps, list) and stored_timestamps:
        selected_timestamps = [
            float(stored_timestamps[index])
            for index in relative_indices
            if index < len(stored_timestamps)
        ]
    else:
        selected_timestamps = []

    metadata = {
        "frame_unit_key": record.get("frame_unit_key"),
        "frame_unit_type": record.get("frame_unit_type"),
        "source_key": record.get("source_key"),
        "source_path": record.get("source_path"),
        "fps": record.get("fps"),
        "total_num_frames": record.get("source_total_num_frames"),
        "source_total_num_frames": record.get("source_total_num_frames"),
        "clip_total_num_frames": record.get("clip_total_num_frames"),
        "stored_frame_count": len(frames),
        "sampled_indices": selected_source_indices,
        "stored_sampled_indices": stored_sampled_indices or [],
        "sampled_timestamps": selected_timestamps,
        "clip_start": record.get("clip_start"),
        "clip_end": record.get("clip_end"),
        "decode_backend": record.get("decode_backend"),
    }
    return images, {key: value for key, value in metadata.items() if value is not None}


def sample_prefixed_preextracted_frame_unit(
    record: Mapping[str, Any],
    *,
    prefix: str,
    num_frames: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Read and sample a frame-unit that was flattened with prefixed columns."""

    prefixed: dict[str, Any] = {}
    prefix_with_separator = f"{prefix}_"
    for key, value in record.items():
        if key.startswith(prefix_with_separator):
            prefixed[key[len(prefix_with_separator):]] = value
    return sample_preextracted_frame_unit(prefixed, num_frames=num_frames)


__all__ = ["decode_frame_bytes", "sample_preextracted_frame_unit", "sample_prefixed_preextracted_frame_unit"]
