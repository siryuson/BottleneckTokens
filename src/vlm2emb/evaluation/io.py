"""Evaluation result serialization and persistence helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def make_serializable(obj: Any) -> Any:
    """Convert non-serializable values to Python primitives."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def build_results_payload(
    results: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize internal benchmark results into a stable save schema."""
    results = make_serializable(results)
    summary = results.get("_summary", {})
    datasets = {
        key: value for key, value in results.items()
        if key != "_summary"
    }
    payload = {
        "meta": make_serializable(meta or {}),
        "summary": make_serializable(summary),
        "datasets": make_serializable(datasets),
    }
    return payload


def save_results_payload(
    output_path: str | os.PathLike,
    results: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Save normalized evaluation results to a single JSON file."""
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_results_payload(results, meta=meta)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return save_path


def save_resolved_config(
    output_path: str | os.PathLike,
    config: Any,
) -> Path:
    """Save resolved config YAML."""
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, dict):
        config = OmegaConf.create(config)
    OmegaConf.save(config=config, f=str(save_path), resolve=True)
    return save_path


def timestamp_slug() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
