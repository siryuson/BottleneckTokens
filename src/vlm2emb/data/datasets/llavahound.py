"""LLaVA-Hound training dataset runtime."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import lance
from PIL import Image

from vlm2emb.auto import AutoDataset
from vlm2emb.data.datasets.base import SampleLanceDataset, SampleTransform
from vlm2emb.data.datasets.const import (
    LEGACY_VIDEO_TOKENS,
    STANDARD_VIDEO_TOKEN,
    canonicalize_multimodal_text,
    decode_image,
    normalize_text_whitespace,
)
from vlm2emb.data.datasets.transforms import (
    join_instruction_body,
    normalize_instruction_body_separator,
    normalize_instruction_text,
    normalize_side_transform_mapping,
)

SUPPORTED_TASK_ARCHETYPES = ("retrieval",)

VRET_QUERY_PROMPT = "Find a video that contains the following visual content: "
VRET_TARGET_PROMPT = "Understand the content of the provided video: "
LlavaHoundMode = Literal["caption_retrieval", "video_retrieval"]
VideoTokenPlacement = Literal["inline_end", "own_line"]
LLAVAHOUND_TRANSFORM_KEYS = {
    "query": {"instruction", "instruction_body_separator", "trailing_newline"},
    "positive": {"instruction", "visual_token_placement", "trailing_newline"},
    "negative": {"empty"},
}
TRAILING_NEWLINE_VALUES = {"preserve", "strip", "ensure_single"}


class LLaVAHoundResolvedVideo(TypedDict):
    """Resolved video payload used by the default training transform.

    Attributes:
        images: Sampled video frames decoded as PIL images, in playback order.
        media_metadata: Runtime hints that describe the source video and the
            sampled frame positions. The training collator treats these values
            as opaque metadata and forwards them with the media slot.
    """

    images: list[Image.Image]
    media_metadata: dict[str, Any]


@dataclass(frozen=True)
class LlavaHoundTransformConfig:
    """Configuration for the built-in LLaVA-Hound sample transform.

    The parameters intentionally mirror the archive parser behavior: caption
    retrieval uses the first conversation turn as the query and the second turn
    as the positive text; video retrieval uses the caption as a text query and
    the sampled frames as the positive target. Instructions are side-scoped so
    YAML config uses the same shape as the other migrated training datasets.
    """

    data_mode: LlavaHoundMode
    query_instruction: str = ""
    query_instruction_body_separator: str = "none"
    query_trailing_newline: str | None = "ensure_single"
    positive_instruction: str = ""
    positive_visual_token_placement: VideoTokenPlacement = "inline_end"
    positive_trailing_newline: str | None = "ensure_single"


def _apply_trailing_newline(text: str, mode: Any, *, dataset_name: str) -> str:
    """Apply one side's trailing-newline policy after text normalization."""

    if mode is None or mode == "preserve":
        return text
    if mode == "strip":
        return text.rstrip("\n")
    if mode == "ensure_single":
        return text.rstrip("\n") + "\n" if text else ""
    raise ValueError(
        f"{dataset_name} trailing_newline must be one of {sorted(TRAILING_NEWLINE_VALUES)}."
    )


def _normalize_transform_kwargs(data_mode: LlavaHoundMode, values: Mapping[str, Any] | None) -> LlavaHoundTransformConfig:
    """Validate and freeze default-transform keyword arguments.

    Args:
        data_mode: Training mode selected by the dataset config.
        values: Mapping supplied through the top-level ``transform`` config
            block or direct constructor keyword arguments.

    Returns:
        A frozen config object consumed by the pickle-safe partial transform.

    Raises:
        ValueError: If unknown transform keys are supplied or if text parameters
            have unsupported types.
    """

    normalized = normalize_side_transform_mapping(
        values,
        dataset_name="LLaVA-Hound",
        allowed_keys=LLAVAHOUND_TRANSFORM_KEYS,
    )
    query = normalized.get("query", {})
    positive = normalized.get("positive", {})
    negative = normalized.get("negative", {})
    visual_token_placement = positive.get("visual_token_placement", "inline_end")
    if visual_token_placement not in {"inline_end", "own_line"}:
        raise ValueError("LLaVA-Hound visual_token_placement must be 'inline_end' or 'own_line'.")
    for side_name, side_values in (("query", query), ("positive", positive)):
        trailing_newline = side_values.get("trailing_newline")
        if trailing_newline is not None and trailing_newline not in TRAILING_NEWLINE_VALUES:
            raise ValueError(
                f"LLaVA-Hound transform.{side_name}.trailing_newline must be one of "
                f"{sorted(TRAILING_NEWLINE_VALUES)}."
            )
    empty = negative.get("empty")
    if empty is not None and empty != "empty_multimodal_input":
        raise ValueError("LLaVA-Hound empty must be 'empty_multimodal_input'.")

    default_query_instruction = VRET_QUERY_PROMPT.rstrip() if data_mode == "video_retrieval" else ""
    default_query_separator = "space" if data_mode == "video_retrieval" else "newline"
    default_positive_instruction = VRET_TARGET_PROMPT.rstrip() if data_mode == "video_retrieval" else ""
    return LlavaHoundTransformConfig(
        data_mode=data_mode,
        query_instruction=normalize_instruction_text(
            query.get("instruction"),
            dataset_name="LLaVA-Hound",
            name="transform.query.instruction",
            default=default_query_instruction,
        ),
        query_instruction_body_separator=normalize_instruction_body_separator(
            query.get("instruction_body_separator"),
            dataset_name="LLaVA-Hound",
            name="transform.query.instruction_body_separator",
            default=default_query_separator,
        ),
        query_trailing_newline=query.get("trailing_newline", "ensure_single"),
        positive_instruction=normalize_instruction_text(
            positive.get("instruction"),
            dataset_name="LLaVA-Hound",
            name="transform.positive.instruction",
            default=default_positive_instruction,
        ),
        positive_visual_token_placement=cast(VideoTokenPlacement, visual_token_placement),
        positive_trailing_newline=positive.get("trailing_newline", "ensure_single"),
    )


def _conversation_value(conversations: Any, index: int) -> str:
    """Read one conversation turn value from the raw instruction row.

    LLaVA-Hound instruction tables store ShareGPT-style ``conversations`` as a
    list of ``{"from": ..., "value": ...}`` objects. Missing or malformed
    turns are normalized to the empty string so the transform can keep archive
    parser behavior while the required video media remains strict.
    """

    if not isinstance(conversations, list) or index >= len(conversations):
        return ""
    turn = conversations[index]
    if not isinstance(turn, Mapping):
        return ""
    value = turn.get("value", "")
    return value if isinstance(value, str) else str(value)


def build_llavahound_caption_query_text(
    conversations: Any,
    *,
    query_instruction: str = "",
    instruction_body_separator: str = "newline",
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the caption-retrieval query text.

    Args:
        conversations: Raw ShareGPT-style conversation list from the instruction
            Lance table.
        query_instruction: Optional instruction prepended before canonicalizing
            the video token. This is used by the QA subset to preserve the archive
            ``llavahound_qa`` prompt.

    Returns:
        Query text with legacy video tokens normalized to
        ``STANDARD_VIDEO_TOKEN`` and moved to a standalone first line when safe.
    """

    query_text = _conversation_value(conversations, 0).replace("<video>", STANDARD_VIDEO_TOKEN)
    query_text = join_instruction_body(
        query_instruction,
        query_text,
        separator=instruction_body_separator,
    )
    text = canonicalize_multimodal_text(
        query_text or STANDARD_VIDEO_TOKEN,
        token=STANDARD_VIDEO_TOKEN,
        legacy_tokens=LEGACY_VIDEO_TOKENS,
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="LLaVA-Hound")


def build_llavahound_caption_positive_text(
    conversations: Any,
    *,
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the caption-retrieval positive text from the assistant turn."""

    text = normalize_text_whitespace(_conversation_value(conversations, 1))
    return _apply_trailing_newline(text, trailing_newline, dataset_name="LLaVA-Hound")


def build_llavahound_video_retrieval_query_text(
    conversations: Any,
    *,
    query_instruction: str = VRET_QUERY_PROMPT.rstrip(),
    instruction_body_separator: str = "space",
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the video-retrieval text query from the caption turn."""

    caption = _conversation_value(conversations, 1)
    text = normalize_text_whitespace(
        join_instruction_body(
            query_instruction,
            caption,
            separator=instruction_body_separator,
        )
    )
    return _apply_trailing_newline(text, trailing_newline, dataset_name="LLaVA-Hound")


def build_llavahound_video_retrieval_positive_text(
    *,
    positive_instruction: str = VRET_TARGET_PROMPT.rstrip(),
    visual_token_placement: VideoTokenPlacement = "inline_end",
    trailing_newline: str | None = "ensure_single",
) -> str:
    """Build the video-retrieval positive text containing the video token."""

    target = positive_instruction.strip()
    if visual_token_placement == "own_line":
        text = canonicalize_multimodal_text(
            f"{STANDARD_VIDEO_TOKEN} {target}" if target else STANDARD_VIDEO_TOKEN,
            token=STANDARD_VIDEO_TOKEN,
            legacy_tokens=LEGACY_VIDEO_TOKENS,
        )
    elif visual_token_placement == "inline_end":
        text = normalize_text_whitespace(
            f"{target} {STANDARD_VIDEO_TOKEN}" if target else STANDARD_VIDEO_TOKEN
        )
    else:
        raise ValueError("LLaVA-Hound visual_token_placement must be 'inline_end' or 'own_line'.")
    return _apply_trailing_newline(text, trailing_newline, dataset_name="LLaVA-Hound")


def build_llavahound_video_media(resolved: LLaVAHoundResolvedVideo) -> list[dict[str, Any]]:
    """Build one video media slot from sampled frames.

    Raises:
        ValueError: If the instruction row cannot be joined to any decoded video
            frames. LLaVA-Hound training samples always require video media, so
            missing frames are treated as conversion/runtime data errors rather
            than silently emitted as token-only samples.
    """

    if not resolved["images"]:
        raise ValueError("LLaVA-Hound row is missing required video frames.")
    return [
        {
            "kind": "video",
            "content": list(resolved["images"]),
            "metadata": dict(resolved["media_metadata"]),
        }
    ]


def transform_llavahound_train_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    config: LlavaHoundTransformConfig,
) -> dict[str, Any]:
    """Transform one resolved instruction row into a ``TrainSample`` dict.

    Args:
        record: Instruction row enriched with ``_resolved_video`` by
            ``LlavaHoundTrainDataset._resolve_records``.
        dataset_name: Stable dataset label recorded in sample metadata.
        config: Built-in transform configuration.

    Returns:
        A schema-shaped training sample with explicit ``query``, ``positive``
        and ``negative`` sides. ``negative`` is intentionally empty because the
        archive parser emits pairwise positive-only samples for these modes.
    """

    conversations = record.get("conversations", [])
    resolved = cast(LLaVAHoundResolvedVideo, record.get("_resolved_video") or _empty_video())
    video_media = build_llavahound_video_media(resolved)
    metadata = {
        key: value
        for key, value in {
            "dataset_name": dataset_name,
            "id": record.get("id"),
            "video": record.get("video"),
            "data_mode": config.data_mode,
        }.items()
        if value is not None
    }

    if config.data_mode == "caption_retrieval":
        return {
            "query": {
                "text": build_llavahound_caption_query_text(
                    conversations,
                    query_instruction=config.query_instruction,
                    instruction_body_separator=config.query_instruction_body_separator,
                    trailing_newline=config.query_trailing_newline,
                ),
                "media": video_media,
            },
            "positive": {
                "text": build_llavahound_caption_positive_text(
                    conversations,
                    trailing_newline=config.positive_trailing_newline,
                ),
                "media": [],
            },
            "negative": {"text": "", "media": []},
            "metadata": metadata,
        }

    return {
        "query": {
            "text": build_llavahound_video_retrieval_query_text(
                conversations,
                query_instruction=config.query_instruction,
                instruction_body_separator=config.query_instruction_body_separator,
                trailing_newline=config.query_trailing_newline,
            ),
            "media": [],
        },
        "positive": {
            "text": build_llavahound_video_retrieval_positive_text(
                positive_instruction=config.positive_instruction,
                visual_token_placement=config.positive_visual_token_placement,
                trailing_newline=config.positive_trailing_newline,
            ),
            "media": video_media,
        },
        "negative": {"text": "", "media": []},
        "metadata": metadata,
    }


def build_llavahound_train_default_transform(
    *,
    dataset_name: str,
    data_mode: LlavaHoundMode,
    transform_kwargs: Mapping[str, Any] | None = None,
) -> SampleTransform:
    """Build the pickle-safe default runtime transform.

    DataLoader workers using ``spawn`` require transform callables to be
    pickleable. Returning a ``functools.partial`` over top-level functions keeps
    the default transform configurable without capturing local closures.
    """

    return partial(
        transform_llavahound_train_sample,
        dataset_name=dataset_name,
        config=_normalize_transform_kwargs(data_mode, transform_kwargs),
    )


def _empty_video() -> LLaVAHoundResolvedVideo:
    """Build an empty video payload used before strict media validation."""

    return {"images": [], "media_metadata": {"total_num_frames": 0, "sampled_indices": []}}


def _video_id_filter(video_id: str) -> str:
    """Build a Lance SQL filter for one raw video id."""

    return "video_id = '" + video_id.replace("'", "''") + "'"


@AutoDataset.register("llavahound_train")
class LlavaHoundTrainDataset(SampleLanceDataset):
    """LLaVA-Hound training dataset over instruction and frame Lance tables.

    The primary table is the instruction table under
    ``data/video_instruction/{subset}.lance``. The frame table is stored once at
    ``data/train_300k.lance`` and joined at runtime by the raw ``video`` field.
    The class intentionally uses ``SampleLanceDataset`` directly: it reads
    instruction rows, resolves sampled video frames in a batch override, then
    applies one final ``SampleTransform`` that returns a ``TrainSample`` dict.
    """

    def __init__(
        self,
        path: str,
        subset: str,
        num_frames: int = 8,
        data_mode: LlavaHoundMode = "caption_retrieval",
        build_index: bool = False,
        dataset_name: str = "llavahound",
        read_columns: list[str] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        transform: SampleTransform | None = None,
        *,
        lazy: bool = True,
    ) -> None:
        if data_mode not in {"caption_retrieval", "video_retrieval"}:
            raise ValueError(f"Unsupported LLaVA-Hound data_mode: {data_mode}")
        if transform is not None and not callable(transform):
            raise TypeError("LlavaHoundTrainDataset transform must be a callable SampleTransform or None")
        if read_columns is not None and columns is not None:
            raise ValueError("Pass only one of read_columns or columns.")
        requested_columns = read_columns if read_columns is not None else columns
        dataset_label = f"{dataset_name}/{subset}"
        effective_transform = (
            transform
            if transform is not None
            else build_llavahound_train_default_transform(
                dataset_name=dataset_label,
                data_mode=data_mode,
                transform_kwargs=transform_kwargs,
            )
        )
        super().__init__(
            lance_path=str(Path(path) / "data" / "video_instruction" / f"{subset}.lance"),
            read_columns=requested_columns,
            metadata=metadata,
            transform=effective_transform,
            lazy=lazy,
        )
        self.path = path
        self.subset = subset
        self.num_frames = num_frames
        self.data_mode = data_mode
        self.build_index = build_index
        self.dataset_name = dataset_label
        self._frames_path = str(Path(path) / "data" / "train_300k.lance")
        self._frames_dataset: lance.LanceDataset | None = None
        self._video_frame_index: dict[str, list[int]] | None = None
        self._has_lance_index: bool | None = None

    @property
    def frames_ds(self) -> lance.LanceDataset:
        """Return the lazily opened shared frame Lance table."""

        if self._frames_dataset is None:
            self._frames_dataset = lance.dataset(self._frames_path)
        return self._frames_dataset

    def _check_lance_index(self) -> bool:
        """Return whether the frame table has a scalar index on ``video_id``."""

        if self._has_lance_index is None:
            indices = self.frames_ds.describe_indices()
            self._has_lance_index = any("video_id" in idx.field_names for idx in indices)
        return self._has_lance_index

    def _build_video_frame_index(self) -> dict[str, list[int]]:
        """Build an in-memory ``video_id`` to frame-row mapping.

        This fallback is only used when the frame table has no scalar index or
        when ``build_index=True`` is requested explicitly. It scans only the key
        columns, then keeps sorted row positions so frame bytes can still be
        fetched lazily for the requested videos.
        """

        if self._video_frame_index is None:
            table = self.frames_ds.to_table(columns=["video_id", "frame_idx"])
            data = table.to_pydict()
            index: dict[str, list[tuple[int, Any]]] = defaultdict(list)
            for row_index, video_id in enumerate(data["video_id"]):
                index[str(video_id)].append((row_index, data["frame_idx"][row_index]))
            self._video_frame_index = {
                video_id: [row_index for row_index, _ in sorted(frames, key=lambda item: self._coerce_frame_position(item[1], 0))]
                for video_id, frames in index.items()
            }
        return self._video_frame_index

    def _sample_indices(self, total: int) -> list[int]:
        """Return deterministic, uniformly spaced frame offsets."""

        if total <= self.num_frames:
            return list(range(total))
        step = total / self.num_frames
        return [int(index * step) for index in range(self.num_frames)]

    def _coerce_frame_position(self, frame_idx: Any, fallback: int) -> int:
        """Convert a raw frame key to a numeric order position when possible."""

        if isinstance(frame_idx, int):
            return frame_idx
        if isinstance(frame_idx, str):
            matches = re.findall(r"\d+", frame_idx)
            if matches:
                return int(matches[-1])
        return fallback

    def _build_resolved_video(
        self,
        frame_pairs: list[tuple[Any, Any]],
    ) -> LLaVAHoundResolvedVideo:
        """Decode sampled frames and attach frame-count metadata."""

        if not frame_pairs:
            return _empty_video()
        sampled_offsets = self._sample_indices(len(frame_pairs))
        sampled_images: list[Image.Image] = []
        sampled_indices: list[int] = []
        for offset in sampled_offsets:
            raw_frame_idx, image_bytes = frame_pairs[offset]
            if image_bytes is None:
                continue
            sampled_images.append(decode_image(image_bytes))
            sampled_indices.append(self._coerce_frame_position(raw_frame_idx, offset))
        return {
            "images": sampled_images,
            "media_metadata": {
                "total_num_frames": len(frame_pairs),
                "sampled_indices": sampled_indices,
            },
        }

    def _get_frames_by_filter(self, video_id: str) -> LLaVAHoundResolvedVideo:
        """Resolve one video's frames using Lance's scalar-index filter path."""

        table = self.frames_ds.to_table(
            columns=["frame_idx", "image"],
            filter=_video_id_filter(video_id),
        )
        if table.num_rows == 0:
            return _empty_video()
        data = table.to_pydict()
        frame_data = sorted(
            zip(data["frame_idx"], data["image"], strict=True),
            key=lambda item: self._coerce_frame_position(item[0], 0),
        )
        return self._build_resolved_video(frame_data)

    def _get_frames_by_index(self, video_id: str) -> LLaVAHoundResolvedVideo:
        """Resolve one video's frames using the in-memory row-position index."""

        index = self._build_video_frame_index()
        if video_id not in index:
            return _empty_video()
        rows = self.frames_ds.take(index[video_id], columns=["frame_idx", "image"]).to_pydict()
        frame_data = list(zip(rows["frame_idx"], rows["image"], strict=True))
        return self._build_resolved_video(frame_data)

    def _get_video_frames(self, video_id: str | None) -> LLaVAHoundResolvedVideo:
        """Resolve sampled frames for one video id."""

        if not video_id:
            return _empty_video()
        if self.build_index:
            return self._get_frames_by_index(video_id)
        if self._check_lance_index():
            return self._get_frames_by_filter(video_id)
        return self._get_frames_by_index(video_id)

    def _resolve_records(self, primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch-resolve frame media for instruction rows.

        The primary instruction table can contain repeated video ids in
        different task views. Resolving each unique id once per batch avoids
        redundant Lance filter calls while preserving the original row order.
        """

        cache: dict[str, LLaVAHoundResolvedVideo] = {}
        resolved_records: list[dict[str, Any]] = []
        for row in primary_rows:
            record = dict(row)
            video_id = cast(str | None, record.get("video"))
            if video_id and video_id not in cache:
                cache[video_id] = self._get_video_frames(video_id)
            record["_resolved_video"] = cache.get(video_id or "", _empty_video())
            resolved_records.append(record)
        return resolved_records

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LlavaHoundTrainDataset:
        """Build a dataset from an ``AutoDataset`` config mapping.

        A mapping-valued ``transform`` block is interpreted as parameters for
        the built-in default transform. A callable ``transform`` remains a full
        replacement for the default transform, matching the new training Dataset
        API used by the other rewritten families.
        """

        kwargs = dict(config)
        kwargs.pop("type", None)
        transform_config = kwargs.get("transform")
        if transform_config is not None and not callable(transform_config):
            if not isinstance(transform_config, Mapping):
                raise TypeError("LLaVA-Hound config transform must be mapping, callable, or None")
            kwargs.pop("transform")
            if "transform_kwargs" in kwargs:
                raise ValueError("LLaVA-Hound config cannot set both transform and transform_kwargs")
            kwargs["transform_kwargs"] = transform_config
        return cls(**kwargs)


__all__ = [
    "SUPPORTED_TASK_ARCHETYPES",
    "VRET_QUERY_PROMPT",
    "VRET_TARGET_PROMPT",
    "LlavaHoundTrainDataset",
    "build_llavahound_caption_positive_text",
    "build_llavahound_caption_query_text",
    "build_llavahound_train_default_transform",
    "build_llavahound_video_media",
    "build_llavahound_video_retrieval_positive_text",
    "build_llavahound_video_retrieval_query_text",
    "transform_llavahound_train_sample",
]
