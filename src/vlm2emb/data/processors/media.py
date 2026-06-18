"""Shared runtime-media parsing helpers for processor wrappers."""

from __future__ import annotations

from typing import Any

from PIL import Image


def _first_present_metadata(
    *candidates: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the first metadata dict that is present, including empty dicts."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def extract_multimodal_payload(
    payload: dict[str, Any],
    *,
    wrapper_name: str,
    context: str,
) -> tuple[str, list[Image.Image], dict[str, Any] | None]:
    """Convert one schema payload into processor-wrapper primitive fields."""
    if not isinstance(payload, dict):
        raise TypeError(f"{context} multimodal payload must be dict")

    text = payload.get("text", "") or ""
    if not isinstance(text, str):
        raise TypeError(f"{context} multimodal text must be str")

    payload_metadata = payload.get("media_metadata")
    if payload_metadata is not None and not isinstance(payload_metadata, dict):
        raise TypeError(f"{context} media_metadata must be dict when present")

    legacy_metadata = payload.get("legacy_media_metadata")
    if legacy_metadata is not None and not isinstance(legacy_metadata, dict):
        raise TypeError(f"{context} legacy_media_metadata must be dict when present")

    if "media" not in payload:
        images = payload.get("images", [])
        if images is None:
            images = []
        if not isinstance(images, list):
            raise TypeError(f"{context} legacy images must be list")
        return text, images, _first_present_metadata(payload_metadata, legacy_metadata)

    if payload.get("images"):
        raise ValueError(f"{context} payload contains both media and legacy images")

    media = payload.get("media") or []
    if not isinstance(media, list):
        raise TypeError(f"{context} multimodal media must be list")

    visual_media: list[Image.Image] = []
    video_metadata: dict[str, Any] | None = None
    seen_video = False
    seen_image = False
    context_label = context.lower()

    for slot in media:
        if not isinstance(slot, dict):
            raise TypeError(f"{context} media slot must be dict")

        kind = slot.get("kind")
        content = slot.get("content")
        if kind == "image":
            if seen_video:
                raise ValueError(
                    f"Mixed image/video media is not supported for {context_label}"
                )
            if not isinstance(content, Image.Image):
                raise TypeError(
                    f"Image {context_label} media content must be PIL.Image at processor boundary"
                )
            visual_media.append(content)
            seen_image = True
            continue

        if kind != "video":
            raise ValueError(f"Unsupported {context_label} media kind: {kind}")

        if seen_image or seen_video:
            raise ValueError(
                f"Mixed image/video media is not supported for {context_label}"
            )
        if not isinstance(content, list):
            raise NotImplementedError(
                f"Raw-video {context_label} media is not supported yet by {wrapper_name}; "
                "use extracted frame-list media for the current public runtime"
            )
        if not all(isinstance(frame, Image.Image) for frame in content):
            raise TypeError(
                f"Video {context_label} media must be list[PIL.Image] when using frame-list mode"
            )

        visual_media = list(content)
        slot_metadata = slot.get("metadata")
        if slot_metadata is not None and not isinstance(slot_metadata, dict):
            raise TypeError(f"{context} video metadata must be dict when present")
        video_metadata = slot_metadata
        seen_video = True

    return text, visual_media, _first_present_metadata(
        payload_metadata,
        video_metadata,
        legacy_metadata,
    )


def extract_retrieval_payload(
    item: dict[str, Any],
    *,
    wrapper_name: str,
) -> tuple[str, list[Image.Image], dict[str, Any] | None]:
    """Extract a query/candidate retrieval item into processor-wrapper fields."""
    payload = item.get("query")
    if payload is None:
        payload = item.get("candidate")
    if not isinstance(payload, dict):
        raise TypeError(
            f"{wrapper_name}.process_retrieval_batch expects "
            "RetrievalQueryItem or RetrievalCandidateItem samples"
        )
    return extract_multimodal_payload(
        payload,
        wrapper_name=wrapper_name,
        context="Retrieval",
    )
