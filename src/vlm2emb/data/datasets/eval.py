"""Retrieval evaluation dataset container.

This module only defines the runtime container used by retrieval evaluators.
MMEB-specific construction now lives in
``vlm2emb.data.datasets.mmeb_v2.loader.build_mmeb_eval_dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm2emb.data.datasets.base import EvalLanceDataset, LanceDataset


@dataclass
class RetrievalEvalDataset:
    """Dataset for retrieval evaluation tasks.
    """

    queries: EvalLanceDataset
    candidates: EvalLanceDataset
    qrels: LanceDataset
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.get("name", "unknown")

    @property
    def task_type(self) -> str | None:
        return self.metadata.get("task_type")

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    def __len__(self) -> int:
        return len(self.queries)

    def __repr__(self) -> str:
        return (
            f"RetrievalEvalDataset("
            f"name={self.name!r}, "
            f"num_queries={self.num_queries}, "
            f"num_candidates={self.num_candidates})"
        )


__all__ = ["RetrievalEvalDataset"]
