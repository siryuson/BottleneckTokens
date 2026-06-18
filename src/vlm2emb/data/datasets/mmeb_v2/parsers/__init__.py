"""Lazy parser module access for MMEB-V2 runtime.

This package intentionally avoids eager importing every parser module during
package initialization. Benchmark-only indexes and runtime dispatch both need
to import parser modules without creating package-level import cycles.

Each parser module is expected to export dataset-local constants plus
`build_query_transform(...)` / `build_candidate_transform(...)` so dispatch can
stay a thin `dataset_name -> parser` router.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_PARSER_MODULE_NAMES = {
    "activitynetqa",
    "didemo",
    "egoschema",
    "image_cls",
    "image_i2i_vg",
    "image_i2t",
    "image_qa",
    "image_t2i",
    "moment_retrieval",
    "momentseeker",
    "msrvtt",
    "msvd",
    "mvbench",
    "nextqa",
    "ssv2",
    "vatex",
    "video_classification",
    "videomme",
    "vidore",
    "visrag",
    "youcook2",
}


def __getattr__(name: str) -> ModuleType:
    if name not in _PARSER_MODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


__all__ = sorted(_PARSER_MODULE_NAMES)
