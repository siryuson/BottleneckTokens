"""Video frame processing utilities.

This module provides utilities for processing video frames:
- Loading frames from directories
- Sampling frames uniformly

Reference:
    BToks project video preprocessing utilities.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

# ===========================================================================
# Constants
# ===========================================================================

VID_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
"""Supported video file extensions."""

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".webp")
"""Supported image file extensions."""

SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES = 0
"""Use sparse seek-based sampling for videos at or above this byte size."""


# ===========================================================================
# Frame loading
# ===========================================================================


def _natural_sort_key(filename: str) -> list[int | str]:
    """Extract numbers from filenames for correct sorting.

    Args:
        filename: Filename to parse.

    Returns:
        List of string/int parts for sorting.

    Example:
        "frame_9.jpg" -> ["frame_", 9, ".jpg"]
        "frame_10.jpg" -> ["frame_", 10, ".jpg"]
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", filename)]


def load_frames(
    frames_dir: str | Path,
    filter_func: Callable[[str], bool] | None = None,
) -> list[str]:
    """Load image frame paths from a directory.

    Frames are sorted using natural sorting (frame_2.jpg < frame_10.jpg).

    Args:
        frames_dir: Directory containing frame images.
        filter_func: Optional function to filter frames by filename.

    Returns:
        List of absolute paths to frame images, sorted naturally.

    Example:
        >>> paths = load_frames("/data/video_001/frames")
        >>> print(paths[:3])
        ['/data/video_001/frames/frame_001.jpg', '/data/video_001/frames/frame_002.jpg', ...]
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []

    results: list[str] = []
    frame_names = sorted(os.listdir(frames_dir), key=_natural_sort_key)

    for frame_name in frame_names:
        ext = os.path.splitext(frame_name)[-1].lower()
        if ext in IMAGE_EXTENSIONS:
            if filter_func is None or filter_func(frame_name):
                image_path = str(frames_dir / frame_name)
                results.append(image_path)

    return results


def sample_frames(frames: list[str], num_segments: int) -> list[str]:
    """Uniformly sample frames from a list.

    Uses numpy linspace to select evenly spaced frame indices.
    If there are fewer frames than requested, the last frame is repeated.

    Args:
        frames: List of frame paths.
        num_segments: Number of frames to sample.

    Returns:
        List of sampled frame paths.

    Example:
        >>> frames = ["frame_001.jpg", "frame_002.jpg", ..., "frame_100.jpg"]
        >>> sampled = sample_frames(frames, num_segments=8)
        >>> len(sampled)
        8
    """
    if not frames:
        return []

    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames: list[str] = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
            sampled_frames.append(single_frame_path)
        except IndexError:
            break

    # If total frame numbers is less than num_segments, repeat last frame.
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])

    return sampled_frames


def process_video_frames(
    frame_dir: str | Path,
    num_frames: int | None = None,
) -> list[str]:
    """Load and sample frames from a frame directory.

    This is the main entry point for video frame processing.
    Combines load_frames() and sample_frames() in one call.

    Args:
        frame_dir: Directory containing extracted frames.
        num_frames: Number of frames to sample. If None, return all frames.
                    If 0, return empty list.

    Returns:
        List of frame paths, uniformly sampled if num_frames specified.

    Example:
        >>> # Load all frames
        >>> all_frames = process_video_frames("/data/video/frames")
        >>> # Sample 8 frames uniformly
        >>> sampled = process_video_frames("/data/video/frames", num_frames=8)
    """
    if num_frames == 0:
        return []

    frames = load_frames(frame_dir)

    if num_frames is None:
        return frames
    elif num_frames <= len(frames):
        return sample_frames(frames, num_segments=num_frames)
    else:
        # Fewer frames than requested - return all and pad if needed.
        return sample_frames(frames, num_segments=num_frames) if frames else []


def probe_video_bytes(video_bytes: bytes, *, source_path: str | Path | None = None) -> dict[str, float | int]:
    """Probe basic timing metadata from raw video bytes."""

    try:
        with av.open(BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            fps, total_num_frames = _probe_pyav_timing(container, stream)
            if total_num_frames <= 0:
                # Last resort for files without usable timing metadata. This can
                # be expensive, so conversion paths should normally avoid it.
                total_num_frames = sum(1 for _ in container.decode(video=0))
        return {
            "fps": fps if fps > 0 else 1.0,
            "total_num_frames": total_num_frames,
        }
    except Exception:
        return _probe_video_bytes_with_cv2(video_bytes, source_path=source_path)


def sample_video_bytes(
    video_bytes: bytes,
    *,
    num_frames: int,
    source_path: str | Path | None = None,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Decode raw video bytes and uniformly sample frames at runtime."""

    if len(video_bytes) >= SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES:
        try:
            return _sample_video_bytes_with_pyav_sparse(video_bytes, num_frames=num_frames)
        except Exception:
            try:
                return _sample_video_bytes_with_cv2(video_bytes, num_frames=num_frames, source_path=source_path)
            except Exception:
                pass

    try:
        with av.open(BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            decoded_frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    except Exception:
        return _sample_video_bytes_with_cv2(video_bytes, num_frames=num_frames, source_path=source_path)

    return _sample_decoded_video_frames(
        decoded_frames,
        fps=fps if fps > 0 else 1.0,
        num_frames=num_frames,
    )


def sample_video_bytes_segment(
    video_bytes: bytes,
    *,
    num_frames: int,
    start_time: float | None = None,
    end_time: float | None = None,
    source_path: str | Path | None = None,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Decode raw video bytes and uniformly sample one temporal segment.

    ``total_num_frames`` and ``sampled_indices`` describe the selected segment,
    not the full source video. Source-video frame context is kept in
    ``source_total_num_frames`` and ``source_sampled_indices``.
    """

    if len(video_bytes) >= SPARSE_VIDEO_SAMPLE_THRESHOLD_BYTES:
        try:
            return _sample_video_bytes_segment_with_pyav_sparse(
                video_bytes,
                num_frames=num_frames,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception:
            try:
                return _sample_video_bytes_segment_with_cv2(
                    video_bytes,
                    num_frames=num_frames,
                    start_time=start_time,
                    end_time=end_time,
                    source_path=source_path,
                )
            except Exception:
                pass

    try:
        with av.open(BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            decoded_frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    except Exception:
        return _sample_video_bytes_segment_with_cv2(
            video_bytes,
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            source_path=source_path,
        )

    return _sample_decoded_video_segment(
        decoded_frames,
        fps=fps if fps > 0 else 1.0,
        num_frames=num_frames,
        start_time=start_time,
        end_time=end_time,
    )


def _probe_pyav_timing(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> tuple[float, int]:
    """Probe basic timing metadata from an opened PyAV video stream."""

    fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
    total_num_frames = int(stream.frames or 0)
    if total_num_frames <= 0 and stream.duration is not None and stream.time_base is not None:
        duration_seconds = float(stream.duration * stream.time_base)
        if fps > 0 and duration_seconds > 0:
            total_num_frames = int(round(duration_seconds * fps))
    if total_num_frames <= 0 and container.duration is not None:
        duration_seconds = float(container.duration / av.time_base)
        if fps > 0 and duration_seconds > 0:
            total_num_frames = int(round(duration_seconds * fps))
    return fps if fps > 0 else 1.0, total_num_frames


def _sample_video_bytes_with_pyav_sparse(
    video_bytes: bytes,
    *,
    num_frames: int,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Sample video frames with PyAV seek operations over in-memory bytes."""

    with av.open(BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        fps, total_num_frames = _probe_pyav_timing(container, stream)
        if total_num_frames <= 0:
            decoded_frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
            return _sample_decoded_video_frames(decoded_frames, fps=fps, num_frames=num_frames)
        sampled_indices = _sample_index_range(
            total_num_frames=total_num_frames,
            start_index=0,
            end_exclusive=total_num_frames,
            num_frames=num_frames,
        )
        sampled_frames, actual_indices = _read_pyav_frame_indices(
            container,
            stream,
            sampled_indices,
            fps=fps,
        )

    if sampled_indices and not sampled_frames:
        raise ValueError("PyAV sparse decode resolved to zero sampled frames.")
    return sampled_frames, {
        "fps": fps,
        "total_num_frames": total_num_frames,
        "sampled_indices": actual_indices,
    }


def _sample_video_bytes_segment_with_pyav_sparse(
    video_bytes: bytes,
    *,
    num_frames: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Sample a temporal segment with PyAV seek operations over in-memory bytes."""

    with av.open(BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        fps, total_num_frames = _probe_pyav_timing(container, stream)
        if total_num_frames <= 0:
            decoded_frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
            return _sample_decoded_video_segment(
                decoded_frames,
                fps=fps,
                num_frames=num_frames,
                start_time=start_time,
                end_time=end_time,
            )
        start_index, end_exclusive = _segment_frame_range(
            total_num_frames=total_num_frames,
            fps=fps,
            start_time=start_time,
            end_time=end_time,
        )
        sampled_indices = _sample_index_range(
            total_num_frames=total_num_frames,
            start_index=start_index,
            end_exclusive=end_exclusive,
            num_frames=num_frames,
        )
        sampled_frames, actual_indices = _read_pyav_frame_indices(
            container,
            stream,
            sampled_indices,
            fps=fps,
        )

    if sampled_indices and not sampled_frames:
        raise ValueError("PyAV sparse segment decode resolved to zero sampled frames.")
    return sampled_frames, {
        "fps": fps,
        "total_num_frames": max(0, end_exclusive - start_index),
        "sampled_indices": [max(0, index - start_index) for index in actual_indices],
        "source_total_num_frames": total_num_frames,
        "source_sampled_indices": actual_indices,
        "segment_start_frame": start_index,
        "segment_end_frame_exclusive": end_exclusive,
    }


def _source_video_suffix(source_path: str | Path | None) -> str:
    """Return a safe temporary suffix that preserves the source container type."""

    if source_path is None:
        return ".mp4"
    suffix = Path(str(source_path)).suffix.lower()
    return suffix if suffix in VID_EXTENSIONS else ".mp4"


def _with_temp_video_file(video_bytes: bytes, *, source_path: str | Path | None = None) -> Path:
    """Persist raw video bytes to one temporary file for fallback decoders."""

    handle = tempfile.NamedTemporaryFile(suffix=_source_video_suffix(source_path), delete=False)
    try:
        handle.write(video_bytes)
        handle.flush()
    finally:
        handle.close()
    return Path(handle.name)


def _probe_video_bytes_with_cv2(
    video_bytes: bytes,
    *,
    source_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Fallback probe path for videos that PyAV cannot parse cleanly."""

    temp_path = _with_temp_video_file(video_bytes, source_path=source_path)
    try:
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV failed to open temporary video: {temp_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_num_frames <= 0:
            total_num_frames = _count_cv2_frames_sequential(capture)
        capture.release()
        return {
            "fps": fps if fps > 0 else 1.0,
            "total_num_frames": total_num_frames,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def _sample_video_bytes_with_cv2(
    video_bytes: bytes,
    *,
    num_frames: int,
    source_path: str | Path | None = None,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Fallback sparse decode path for videos that PyAV cannot parse cleanly."""

    temp_path = _with_temp_video_file(video_bytes, source_path=source_path)
    try:
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV failed to open temporary video: {temp_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_num_frames <= 0:
            decoded_frames = _read_cv2_frames_sequential(capture)
            return _sample_decoded_video_frames(
                decoded_frames,
                fps=fps if fps > 0 else 1.0,
                num_frames=num_frames,
            )
        sampled_indices = _sample_index_range(
            total_num_frames=total_num_frames,
            start_index=0,
            end_exclusive=total_num_frames,
            num_frames=num_frames,
        )
        sampled_frames, actual_indices = _read_cv2_frame_indices(
            capture,
            sampled_indices,
            total_num_frames=total_num_frames,
        )
    finally:
        try:
            capture.release()
        except UnboundLocalError:
            pass
        temp_path.unlink(missing_ok=True)

    if total_num_frames == 0:
        return [], {"fps": fps if fps > 0 else 1.0, "total_num_frames": 0, "sampled_indices": []}

    return sampled_frames, {
        "fps": fps if fps > 0 else 1.0,
        "total_num_frames": total_num_frames,
        "sampled_indices": actual_indices,
    }


def _sample_video_bytes_segment_with_cv2(
    video_bytes: bytes,
    *,
    num_frames: int,
    start_time: float | None = None,
    end_time: float | None = None,
    source_path: str | Path | None = None,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Fallback sparse segment decode path for videos that PyAV cannot parse cleanly."""

    temp_path = _with_temp_video_file(video_bytes, source_path=source_path)
    try:
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV failed to open temporary video: {temp_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        safe_fps = fps if fps > 0 else 1.0
        total_num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_num_frames <= 0:
            decoded_frames = _read_cv2_frames_sequential(capture)
            return _sample_decoded_video_segment(
                decoded_frames,
                fps=safe_fps,
                num_frames=num_frames,
                start_time=start_time,
                end_time=end_time,
            )
        start_index, end_exclusive = _segment_frame_range(
            total_num_frames=total_num_frames,
            fps=safe_fps,
            start_time=start_time,
            end_time=end_time,
        )
        sampled_indices = _sample_index_range(
            total_num_frames=total_num_frames,
            start_index=start_index,
            end_exclusive=end_exclusive,
            num_frames=num_frames,
        )
        sampled_frames, actual_indices = _read_cv2_frame_indices(
            capture,
            sampled_indices,
            total_num_frames=total_num_frames,
        )
    finally:
        try:
            capture.release()
        except UnboundLocalError:
            pass
        temp_path.unlink(missing_ok=True)

    return sampled_frames, {
        "fps": fps if fps > 0 else 1.0,
        "total_num_frames": max(0, end_exclusive - start_index),
        "sampled_indices": [max(0, index - start_index) for index in actual_indices],
        "source_total_num_frames": total_num_frames,
        "source_sampled_indices": actual_indices,
        "segment_start_frame": start_index,
        "segment_end_frame_exclusive": end_exclusive,
    }


def _sample_index_range(
    *,
    total_num_frames: int,
    start_index: int,
    end_exclusive: int,
    num_frames: int,
) -> list[int]:
    """Return frame indices to sample from one bounded source range."""

    if total_num_frames <= 0 or num_frames <= 0:
        return []
    start_index = max(0, min(start_index, total_num_frames - 1))
    end_exclusive = max(start_index + 1, min(end_exclusive, total_num_frames))
    span = end_exclusive - start_index
    if span <= num_frames:
        return list(range(start_index, end_exclusive))
    return np.linspace(start_index, end_exclusive - 1, num_frames, dtype=int).tolist()


def _sample_decoded_video_frames(
    decoded_frames: list[Image.Image],
    *,
    fps: float,
    num_frames: int,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Uniformly sample already-decoded full-video frames."""

    total_num_frames = len(decoded_frames)
    if total_num_frames == 0:
        return [], {"fps": fps, "total_num_frames": 0, "sampled_indices": []}

    if num_frames <= 0:
        sampled_indices: list[int] = []
        sampled_frames: list[Image.Image] = []
    elif total_num_frames <= num_frames:
        sampled_indices = list(range(total_num_frames))
        sampled_frames = decoded_frames
    else:
        sampled_indices = np.linspace(0, total_num_frames - 1, num_frames, dtype=int).tolist()
        sampled_frames = [decoded_frames[index] for index in sampled_indices]

    return sampled_frames, {
        "fps": fps,
        "total_num_frames": total_num_frames,
        "sampled_indices": sampled_indices,
    }


def _segment_frame_range(
    *,
    total_num_frames: int,
    fps: float,
    start_time: float | None,
    end_time: float | None,
) -> tuple[int, int]:
    """Convert an optional time segment into source frame bounds."""

    if total_num_frames <= 0:
        return 0, 0
    if start_time is None and end_time is None:
        return 0, total_num_frames
    safe_start = max(0.0, float(start_time or 0.0))
    safe_end = float(end_time) if end_time is not None else total_num_frames / fps
    safe_end = max(safe_start, safe_end)
    start_index = min(total_num_frames - 1, int(safe_start * fps))
    end_exclusive = min(total_num_frames, max(start_index + 1, int(np.ceil(safe_end * fps))))
    return start_index, end_exclusive


def _read_pyav_frame_indices(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    sampled_indices: list[int],
    *,
    fps: float,
) -> tuple[list[Image.Image], list[int]]:
    """Read selected frame indices from an opened PyAV container."""

    if stream.time_base is None:
        raise ValueError("PyAV stream is missing time_base for sparse frame seek.")
    sampled_frames: list[Image.Image] = []
    actual_indices: list[int] = []
    for index in sampled_indices:
        target_seconds = index / fps
        target_pts = int(target_seconds / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            actual_index = _pyav_frame_index(frame, fps=fps)
            if actual_index is None:
                sampled_frames.append(frame.to_image().convert("RGB"))
                actual_indices.append(index)
                break
            if actual_index >= index:
                sampled_frames.append(frame.to_image().convert("RGB"))
                actual_indices.append(actual_index)
                break
    return sampled_frames, actual_indices


def _pyav_frame_index(frame: av.video.frame.VideoFrame, *, fps: float) -> int | None:
    """Infer a source frame index from PyAV timing metadata."""

    if frame.pts is None or frame.time_base is None:
        return None
    return max(0, int(round(float(frame.pts * frame.time_base) * fps)))


def _read_cv2_frame_indices(
    capture: cv2.VideoCapture,
    sampled_indices: list[int],
    *,
    total_num_frames: int,
) -> tuple[list[Image.Image], list[int]]:
    """Read selected frame indices from an opened OpenCV capture."""

    sampled_frames: list[Image.Image] = []
    actual_indices: list[int] = []
    for index in sampled_indices:
        success = False
        frame = None
        actual_index = index
        for candidate_index in _nearby_frame_indices(index, total_num_frames=total_num_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
            success, frame = capture.read()
            if success:
                actual_index = candidate_index
                break
        if not success or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled_frames.append(Image.fromarray(rgb))
        actual_indices.append(actual_index)
    return sampled_frames, actual_indices


def _read_cv2_frames_sequential(capture: cv2.VideoCapture) -> list[Image.Image]:
    """Decode all frames sequentially when random-access metadata is unavailable."""

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames: list[Image.Image] = []
    while True:
        success, frame = capture.read()
        if not success or frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    return frames


def _count_cv2_frames_sequential(capture: cv2.VideoCapture) -> int:
    """Count frames sequentially when OpenCV cannot report frame count."""

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        success, frame = capture.read()
        if not success or frame is None:
            break
        count += 1
    return count


def _nearby_frame_indices(index: int, *, total_num_frames: int) -> list[int]:
    """Return a small fallback window for codecs with imprecise random seek."""

    if total_num_frames <= 0:
        return []
    max_index = total_num_frames - 1
    bounded_index = max(0, min(index, max_index))
    candidates = [bounded_index]
    for offset in range(1, 8):
        lower = bounded_index - offset
        upper = bounded_index + offset
        if lower >= 0:
            candidates.append(lower)
        if upper <= max_index:
            candidates.append(upper)
    return candidates


def _sample_decoded_video_segment(
    decoded_frames: list[Image.Image],
    *,
    fps: float,
    num_frames: int,
    start_time: float | None,
    end_time: float | None,
) -> tuple[list[Image.Image], dict[str, float | int | list[int]]]:
    """Uniformly sample one temporal segment from decoded video frames."""

    total_num_frames = len(decoded_frames)
    if total_num_frames == 0:
        return [], {
            "fps": fps,
            "total_num_frames": 0,
            "sampled_indices": [],
            "source_total_num_frames": 0,
            "source_sampled_indices": [],
            "segment_start_frame": 0,
            "segment_end_frame_exclusive": 0,
        }

    if start_time is None and end_time is None:
        start_index = 0
        end_exclusive = total_num_frames
    else:
        safe_start = max(0.0, float(start_time or 0.0))
        safe_end = float(end_time) if end_time is not None else total_num_frames / fps
        safe_end = max(safe_start, safe_end)
        start_index = min(total_num_frames - 1, max(0, int(np.floor(safe_start * fps))))
        end_exclusive = max(start_index + 1, int(np.ceil(safe_end * fps)))
        end_exclusive = min(total_num_frames, end_exclusive)

    segment_indices = list(range(start_index, end_exclusive))
    if not segment_indices:
        segment_indices = [min(total_num_frames - 1, start_index)]
    segment_total_num_frames = len(segment_indices)

    if num_frames <= 0:
        sampled_indices: list[int] = []
        sampled_frames: list[Image.Image] = []
    elif len(segment_indices) <= num_frames:
        sampled_indices = segment_indices
        sampled_frames = [decoded_frames[index] for index in sampled_indices]
    else:
        relative_indices = np.linspace(0, len(segment_indices) - 1, num_frames, dtype=int).tolist()
        sampled_indices = [segment_indices[index] for index in relative_indices]
        sampled_frames = [decoded_frames[index] for index in sampled_indices]

    return sampled_frames, {
        "fps": fps,
        "total_num_frames": segment_total_num_frames,
        "sampled_indices": [index - start_index for index in sampled_indices],
        "source_total_num_frames": total_num_frames,
        "source_sampled_indices": sampled_indices,
        "segment_start_frame": start_index,
        "segment_end_frame_exclusive": end_exclusive,
    }
