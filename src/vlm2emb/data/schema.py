"""Public runtime schema definitions for the data subsystem.

These TypedDicts define the model-facing runtime contract shared across
datasets, collators, and processors. They are intentionally stored at the
``vlm2emb.data`` package level because they are not specific to dataset
implementations.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from PIL import Image

MediaKind = Literal["image", "video"]


class MediaInput(TypedDict, total=False):
    """One ordered media slot inside one multimodal model input.

    ``content`` prefers already-prepared runtime objects:
    - image -> one ``PIL.Image``
    - video -> ``list[PIL.Image]`` or one direct video object

    ``metadata`` is optional per-slot metadata. It is mainly useful for video
    timing information such as fps or sampled frame indices.

    For video inputs, both representations are first-class schema options.
    Project guidance may recommend already-prepared frame lists for throughput,
    but this recommendation does not change the logical status of either form.
    """

    kind: MediaKind
    content: Image.Image | list[Image.Image] | Any
    metadata: dict[str, Any]


class MultiModalInput(TypedDict):
    """Unified model-facing multimodal input."""

    text: str
    media: list[MediaInput]


class TrainSample(TypedDict, total=False):
    """Runtime training sample built from multimodal side inputs."""

    query: MultiModalInput
    positive: MultiModalInput
    negative: MultiModalInput
    metadata: dict[str, Any]


class RetrievalQueryItem(TypedDict, total=False):
    """One retrieval query item."""

    id: str
    query: MultiModalInput
    metadata: dict[str, Any]


class RetrievalCandidateItem(TypedDict, total=False):
    """One retrieval candidate item."""

    id: str
    candidate: MultiModalInput
    metadata: dict[str, Any]


class QrelsRow(TypedDict):
    """One retrieval relevance row."""

    query_id: str
    mode: str
    candidate_ids: list[str]
    candidate_scores: list[float]
