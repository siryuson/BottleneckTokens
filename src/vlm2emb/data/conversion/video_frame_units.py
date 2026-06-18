"""Frame-unit conversion helpers for video training datasets."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from PIL import Image

from vlm2emb.data.utils.video import sample_video_bytes, sample_video_bytes_segment

FULL_VIDEO_MAX_FRAMES = 64
SEGMENT_MAX_FRAMES = 8

FRAME_BYTES_TYPE = pa.list_(pa.binary())
INT_LIST_TYPE = pa.list_(pa.int32())
FLOAT_LIST_TYPE = pa.list_(pa.float64())

FRAME_UNIT_SCHEMA = pa.schema(
    [
        pa.field("frame_unit_key", pa.string()),
        pa.field("frame_unit_type", pa.string()),
        pa.field("source_key", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("frames", FRAME_BYTES_TYPE),
        pa.field("sampled_frame_count", pa.int32()),
        pa.field("source_total_num_frames", pa.int32()),
        pa.field("clip_total_num_frames", pa.int32()),
        pa.field("fps", pa.float32()),
        pa.field("duration", pa.float64()),
        pa.field("clip_start", pa.float64()),
        pa.field("clip_end", pa.float64()),
        pa.field("sampled_indices", INT_LIST_TYPE),
        pa.field("sampled_timestamps", FLOAT_LIST_TYPE),
        pa.field("decode_backend", pa.string()),
        pa.field("warning_count", pa.int32()),
    ]
)

BAD_MEDIA_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string()),
        pa.field("split", pa.string()),
        pa.field("source_key", pa.string()),
        pa.field("frame_unit_key", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("error", pa.string()),
        pa.field("source_path", pa.string()),
    ]
)


def sample_indices(total: int, max_frames: int) -> list[int]:
    """Return uniform indices without padding or repetition."""

    if total <= 0 or max_frames <= 0:
        return []
    if total <= max_frames:
        return list(range(total))
    return np.linspace(0, total - 1, max_frames, dtype=int).tolist()


def encode_image_to_jpeg(image: Image.Image, *, quality: int = 90) -> bytes:
    """Encode one image as RGB JPEG bytes."""

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def encode_images_to_jpeg(images: list[Image.Image], *, quality: int = 90) -> list[bytes]:
    """Encode a list of images as RGB JPEG bytes."""

    return [encode_image_to_jpeg(image, quality=quality) for image in images]


def _timestamps_for_indices(indices: list[int], *, fps: float) -> list[float]:
    """Convert frame indices to timestamps using a positive fps."""

    safe_fps = fps if fps > 0 else 1.0
    return [float(index) / safe_fps for index in indices]


def build_bad_media_row(
    *,
    dataset: str,
    split: str,
    source_key: str,
    frame_unit_key: str,
    reason: str,
    error: str,
    source_path: str,
) -> dict[str, Any]:
    """Build one bad-media audit row."""

    return {
        "dataset": dataset,
        "split": split,
        "source_key": source_key,
        "frame_unit_key": frame_unit_key,
        "reason": reason,
        "error": error,
        "source_path": source_path,
    }


def build_frame_unit_from_images(
    *,
    frame_unit_key: str,
    frame_unit_type: str,
    source_key: str,
    source_path: str,
    images: list[Image.Image],
    source_total_num_frames: int,
    sampled_indices: list[int],
    fps: float = 0.0,
    duration: float = 0.0,
    clip_start: float = 0.0,
    clip_end: float = 0.0,
    clip_total_num_frames: int | None = None,
    decode_backend: str,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Build one frame-unit row from sampled images."""

    if not images:
        raise ValueError(f"Frame unit resolved to zero frames: {frame_unit_key}")
    safe_source_total = max(0, int(source_total_num_frames))
    safe_clip_total = (
        max(0, int(clip_total_num_frames))
        if clip_total_num_frames is not None
        else safe_source_total
    )
    safe_fps = float(fps) if fps and fps > 0 else 0.0
    return {
        "frame_unit_key": frame_unit_key,
        "frame_unit_type": frame_unit_type,
        "source_key": source_key,
        "source_path": source_path,
        "frames": encode_images_to_jpeg(images, quality=jpeg_quality),
        "sampled_frame_count": len(images),
        "source_total_num_frames": safe_source_total,
        "clip_total_num_frames": safe_clip_total,
        "fps": safe_fps,
        "duration": float(duration) if duration else 0.0,
        "clip_start": float(clip_start) if clip_start else 0.0,
        "clip_end": float(clip_end) if clip_end else 0.0,
        "sampled_indices": [int(index) for index in sampled_indices],
        "sampled_timestamps": _timestamps_for_indices(sampled_indices, fps=safe_fps),
        "decode_backend": decode_backend,
        "warning_count": 0,
    }


def build_full_video_frame_unit_from_bytes(
    *,
    video_bytes: bytes,
    frame_unit_key: str,
    source_key: str,
    source_path: str | Path,
    max_frames: int = FULL_VIDEO_MAX_FRAMES,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Sample one full video or pre-clipped video into a frame-unit row."""

    images, metadata = sample_video_bytes(
        video_bytes,
        num_frames=max_frames,
        source_path=source_path,
    )
    return build_frame_unit_from_images(
        frame_unit_key=frame_unit_key,
        frame_unit_type="full_video",
        source_key=source_key,
        source_path=str(source_path),
        images=images,
        source_total_num_frames=int(metadata.get("total_num_frames", len(images))),
        sampled_indices=list(metadata.get("sampled_indices", [])),
        fps=float(metadata.get("fps", 0.0) or 0.0),
        duration=(
            float(metadata.get("total_num_frames", 0) or 0)
            / float(metadata.get("fps", 1.0) or 1.0)
        ),
        decode_backend="video_bytes",
        jpeg_quality=jpeg_quality,
    )


def build_segment_frame_unit_from_bytes(
    *,
    video_bytes: bytes,
    frame_unit_key: str,
    source_key: str,
    source_path: str | Path,
    start_time: float,
    end_time: float,
    max_frames: int = SEGMENT_MAX_FRAMES,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Sample one temporal segment into a frame-unit row."""

    images, metadata = sample_video_bytes_segment(
        video_bytes,
        num_frames=max_frames,
        start_time=start_time,
        end_time=end_time,
        source_path=source_path,
    )
    return build_frame_unit_from_images(
        frame_unit_key=frame_unit_key,
        frame_unit_type="segment",
        source_key=source_key,
        source_path=str(source_path),
        images=images,
        source_total_num_frames=int(metadata.get("source_total_num_frames", metadata.get("total_num_frames", 0))),
        sampled_indices=list(metadata.get("source_sampled_indices", metadata.get("sampled_indices", []))),
        fps=float(metadata.get("fps", 0.0) or 0.0),
        clip_start=start_time,
        clip_end=end_time,
        clip_total_num_frames=int(metadata.get("total_num_frames", len(images))),
        decode_backend="video_bytes_segment",
        jpeg_quality=jpeg_quality,
    )


def build_row_frame_unit_from_rgb_bytes(
    *,
    frame_unit_key: str,
    source_key: str,
    source_path: str,
    binary_frames: bytes,
    num_frames: int,
    height: int,
    width: int,
    channels: int,
    max_frames: int = SEGMENT_MAX_FRAMES,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Build a row-level segment frame-unit from contiguous uint8 RGB frames."""

    total_frames = int(num_frames)
    frame_height = int(height)
    frame_width = int(width)
    frame_channels = int(channels)
    if total_frames <= 0:
        raise ValueError("row-level frame tensor has no frames")
    if frame_height <= 0 or frame_width <= 0 or frame_channels not in {1, 3, 4}:
        raise ValueError(
            "row-level frame tensor has invalid shape: "
            f"num_frames={total_frames}, height={frame_height}, width={frame_width}, channels={frame_channels}"
        )
    expected_size = total_frames * frame_height * frame_width * frame_channels
    if len(binary_frames) != expected_size:
        raise ValueError(f"row-level frame tensor bytes={len(binary_frames)} expected={expected_size}")

    array = np.frombuffer(binary_frames, dtype=np.uint8).reshape(
        total_frames,
        frame_height,
        frame_width,
        frame_channels,
    )
    indices = sample_indices(total_frames, max_frames)
    images: list[Image.Image] = []
    for index in indices:
        frame_array = array[index]
        if frame_channels == 1:
            image = Image.fromarray(frame_array[:, :, 0]).convert("RGB")
        else:
            image = Image.fromarray(frame_array[:, :, :3])
        images.append(image)
    return build_frame_unit_from_images(
        frame_unit_key=frame_unit_key,
        frame_unit_type="row_segment",
        source_key=source_key,
        source_path=source_path,
        images=images,
        source_total_num_frames=total_frames,
        sampled_indices=indices,
        fps=0.0,
        clip_total_num_frames=total_frames,
        decode_backend="row_rgb_tensor",
        jpeg_quality=jpeg_quality,
    )


__all__ = [
    "BAD_MEDIA_SCHEMA",
    "FRAME_UNIT_SCHEMA",
    "FULL_VIDEO_MAX_FRAMES",
    "SEGMENT_MAX_FRAMES",
    "build_bad_media_row",
    "build_frame_unit_from_images",
    "build_full_video_frame_unit_from_bytes",
    "build_row_frame_unit_from_rgb_bytes",
    "build_segment_frame_unit_from_bytes",
    "encode_image_to_jpeg",
    "sample_indices",
]
