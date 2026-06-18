"""Shared MMEB parser helpers.

This module only contains low-level helpers reused by multiple parser files.
It must stay free of dataset-specific routing so the runtime architecture
remains `dataset -> parser -> query/candidate transform`.

Parser public entrypoints use build_query_transform/build_candidate_transform;
the compose_* helpers describe text assembly internals, not a separate runtime
contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm2emb.data.datasets.const import (
    LEGACY_IMAGE_TOKENS,
    STANDARD_IMAGE_TOKEN,
    STANDARD_VIDEO_TOKEN,
    normalize_text_whitespace,
)
from vlm2emb.data.schema import MediaInput


def process_input_text(
    instruction: str,
    *,
    text: str | None = None,
    add_image_token: bool = False,
    add_video_token: bool = False,
    token_separator: str = ' ',
) -> str:
    """Compose one prompt string before parser-specific newline handling."""
    prompt = instruction
    if text:
        prompt = prompt + ' ' + text
    if add_video_token:
        prompt = STANDARD_VIDEO_TOKEN + token_separator + prompt
    if add_image_token:
        prompt = STANDARD_IMAGE_TOKEN + token_separator + prompt
    return prompt


def format_choice_template(question: str, candidates: list[str]) -> str:
    """Format archive-style option text with `(A)/(B)/...` labels."""
    composed = f"{question}\n"
    composed += "Options:\n"
    for index, candidate in enumerate(candidates):
        composed += f"({chr(ord('A') + index)}) {candidate}\n"
    return composed.rstrip()


def replace_legacy_image_tokens(text: str, replacement: str = STANDARD_IMAGE_TOKEN) -> str:
    normalized = text
    for token in LEGACY_IMAGE_TOKENS:
        normalized = normalized.replace(token, replacement)
    return normalized


def strip_legacy_image_tokens(text: str) -> str:
    normalized = text
    for token in LEGACY_IMAGE_TOKENS:
        normalized = normalized.replace(token, '')
    normalized = normalize_text_whitespace(normalized)
    return normalized.lstrip('\n')


def compose_parser_text_with_instruction(
    text: str,
    *,
    instruction: str,
    add_image_token: bool = False,
    add_video_token: bool = False,
    token_separator: str = ' ',
    replace_space_newline: bool = False,
    append_trailing_newline: bool = False,
    append_extra_blank_line: bool = False,
) -> str:
    """Build parser text for legacy MMEB-V2 eval prompt compatibility."""
    composed = process_input_text(
        instruction,
        text=text or None,
        add_image_token=add_image_token,
        add_video_token=add_video_token,
        token_separator=token_separator,
    )
    if replace_space_newline:
        composed = composed.replace(' \n', '\n')
    if append_trailing_newline:
        composed = composed + '\n'
    if append_extra_blank_line:
        composed = composed + '\n'
    return composed


def extract_item_metadata(sample: dict[str, Any], *, consumed_fields: set[str]) -> dict[str, Any]:
    """Keep untouched raw fields as runtime metadata for debugging and audits."""
    return {
        key: value
        for key, value in sample.items()
        if key not in consumed_fields and value is not None
    }


def collect_image_media(sample: dict[str, Any]) -> list[MediaInput]:
    return [
        {'kind': 'image', 'content': image}
        for image in (sample.get('images') or [])
        if image is not None
    ]


def take_video_frames(
    sample: dict[str, Any],
    *,
    defaults: dict[str, Any],
    frame_key: str = 'num_frames',
    metadata_key: str = 'sampled_indices',
) -> tuple[list[Any], dict[str, Any]]:
    """Sample frames according to parser defaults and preserve sampling metadata.

    Runtime uses deterministic uniform sampling here on purpose: eval prompts
    must stay reproducible across runs, workers, and audit sessions. We also
    write `total_num_frames` plus the sampled indices back into media metadata
    so downstream debugging can explain exactly which visual evidence the model
    saw for one composed sample.
    """
    all_frames = sample.get('video') or sample.get('images') or []
    frames = [frame for frame in all_frames if frame is not None]
    if not frames:
        return [], dict(sample.get('media_metadata') or {})
    target_frames = defaults.get(frame_key)
    try:
        target_frames_int = int(target_frames) if target_frames is not None else None
    except (TypeError, ValueError):
        target_frames_int = None
    if target_frames_int is not None and target_frames_int > 0 and len(frames) > target_frames_int:
        sampled_indices = np.linspace(0, len(frames) - 1, target_frames_int, dtype=int).tolist()
        frames = [frames[index] for index in sampled_indices]
    else:
        sampled_indices = list(range(len(frames)))
    metadata = dict(sample.get('media_metadata') or {})
    metadata.setdefault('total_num_frames', len(all_frames))
    metadata[metadata_key] = sampled_indices
    return frames, metadata


def collect_video_media(
    sample: dict[str, Any],
    *,
    defaults: dict[str, Any],
    frame_key: str = 'num_frames',
    metadata_key: str = 'sampled_indices',
) -> list[MediaInput]:
    frames, metadata = take_video_frames(
        sample,
        defaults=defaults,
        frame_key=frame_key,
        metadata_key=metadata_key,
    )
    if not frames:
        return []
    media: MediaInput = {'kind': 'video', 'content': frames}
    if metadata:
        media['metadata'] = metadata
    return [media]


def require_non_empty_string(
    sample: dict[str, Any],
    field_name: str,
    *,
    dataset_name: str,
    task_type: str,
    role: str,
) -> str:
    value = sample.get(field_name)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(
        f'Missing required raw field for MMEB eval transform: '
        f'dataset_name={dataset_name}, task_type={task_type}, role={role}, field={field_name}'
    )
