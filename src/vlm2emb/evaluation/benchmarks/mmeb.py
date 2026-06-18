"""MMEB benchmark runtime.

This module defines the benchmark entry used to evaluate MMEB-V2 subsets
through the parser-driven runtime package under
``vlm2emb.data.datasets.mmeb_v2``.

Responsibilities in this file stay intentionally narrow:
- resolve the fixed benchmark subset set from the static benchmark index;
- validate that on-disk subset artifacts are complete;
- construct one :class:`RetrievalEvaluator` per discovered subset;
- aggregate per-subset scores into task-type, modality, and overall summaries.

The benchmark does not own parser-specific text-transform logic. The retrieval
container comes from :mod:`vlm2emb.data.datasets.eval`, while query/candidate
transforms are provided by the ``vlm2emb.data.datasets.mmeb_v2`` runtime
package.
"""

from __future__ import annotations

import os
from typing import Any, cast

from vlm2emb.auto import AutoBenchmark
from vlm2emb.data.datasets.mmeb_v2 import build_mmeb_eval_dataset
from vlm2emb.data.datasets.mmeb_v2.benchmark_index import (
    BENCHMARK_INDEX_ROWS,
    DATASETS_BY_MODALITY,
    MODALITY_BY_DATASET,
    PATH_REL_BY_DATASET,
    TASK_TYPE_BY_DATASET,
)
from vlm2emb.evaluation.benchmarks.base import BaseBenchmark
from vlm2emb.evaluation.evaluators.base import BaseEvaluator
from vlm2emb.evaluation.evaluators.retrieval import RetrievalEvaluator
from vlm2emb.evaluation.metrics import DEFAULT_RETRIEVAL_METRICS
from vlm2emb.utils.logging import RankLogger

logger = RankLogger(__name__)

MODALITY_METRIC: dict[str, str] = {
    "image": "hit@1",
    "video": "hit@1",
    "visdoc": "ndcg@5",
}
"""Primary aggregation metric used for each modality."""

MMEB_RUNTIME_REGISTRY = {name: path_rel for name, path_rel, _, _ in BENCHMARK_INDEX_ROWS}
"""Static mapping from subset name to benchmark artifact relative path."""

DATASET_REGISTRY = DATASETS_BY_MODALITY
"""Static subset listing grouped by modality."""


@AutoBenchmark.register("mmeb")
class MMEBBenchmark(BaseBenchmark):
    """Benchmark wrapper for MMEB multimodal retrieval evaluation.

    One benchmark instance represents one MMEB evaluation job rooted at
    ``data_path``. The instance remains lightweight during construction:
    subset discovery and evaluator creation are deferred until the
    :attr:`evaluators` property is first accessed.
    """

    def __init__(
        self,
        data_path: str,
        datasets: list[str] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        num_frames: int = 8,
        metrics: list[str] | tuple[str, ...] | None = None,
        show_progress: bool = True,
        runtime_mode: str = "canonical",
    ):
        """Initialize one MMEB benchmark job.

        Args:
            data_path: Root directory of the converted MMEB-V2 benchmark
                artifacts. The benchmark expects the benchmark-index task
                layout under this root, for example ``image-tasks/VisDial``.
            datasets: Optional subset allowlist. When provided, only subset
                names present in this list are considered during discovery.
            batch_size: Encoding batch size passed to every retrieval evaluator.
            num_workers: Number of dataloader workers passed to every evaluator.
                MMEB creates many short-lived query/candidate dataloaders, so
                the benchmark defaults to ``0`` to avoid paying repeated
                spawn-based cold-start costs on every subset.
            num_frames: Default frame budget forwarded to the retrieval runtime.
                This mainly affects video-family parsers that sample frame lists.
            metrics: Complete metric names requested from each retrieval evaluator.
            show_progress: Whether evaluator-side encoding loops should display
                progress bars on the main process.
            runtime_mode: MMEB runtime transform mode. ``canonical`` is the default
                project mode; ``origin`` preserves original parser quirks for
                alignment and review.
        """
        super().__init__(name="MMEB")
        self.data_path = os.path.expanduser(data_path)
        self.dataset_names = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.metrics = tuple(metrics) if metrics is not None else DEFAULT_RETRIEVAL_METRICS
        self.show_progress = show_progress
        self.runtime_mode = runtime_mode
        self._dataset_meta: dict[str, dict[str, Any]] = {}

    @property
    def evaluators(self) -> list[BaseEvaluator]:
        """Lazily build the retrieval evaluators for discovered subsets."""
        if self._evaluators is None:
            self._evaluators = cast(list[BaseEvaluator], self._create_evaluators())
        return self._evaluators

    def _create_evaluators(self) -> list[RetrievalEvaluator]:
        """Create one retrieval evaluator per expected MMEB subset.

        Returns:
            A list of evaluators, one for each validated MMEB subset.
        """
        evaluators = []
        self._resolve_datasets()
        self._validate_metrics_cover_aggregation()

        for dataset_name, meta in self._dataset_meta.items():
            dataset = build_mmeb_eval_dataset(
                dataset_name=dataset_name,
                artifact_path=cast(str, meta["path"]),
                transform_kwargs={
                    "num_frames": self.num_frames,
                    "runtime_mode": self.runtime_mode,
                },
            )
            evaluators.append(
                RetrievalEvaluator(
                    dataset=dataset,
                    metrics=self.metrics,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    show_progress=self.show_progress,
                )
            )

        return evaluators

    def _required_aggregation_metrics(self) -> tuple[str, ...]:
        """Return modality-specific metrics needed by benchmark aggregation."""
        required: list[str] = []
        for modality in ("image", "video", "visdoc"):
            if any(meta["modality"] == modality for meta in self._dataset_meta.values()):
                metric = MODALITY_METRIC.get(modality, "hit@1")
                if metric not in required:
                    required.append(metric)
        return tuple(required)

    def _validate_metrics_cover_aggregation(self) -> None:
        """Ensure custom metric requests still support MMEB summary aggregation."""
        missing = [metric for metric in self._required_aggregation_metrics() if metric not in self.metrics]
        if missing:
            raise ValueError(
                "MMEB metrics must include aggregation metric(s): "
                f"{', '.join(missing)}. Requested metrics: {', '.join(self.metrics)}"
            )

    def _resolve_expected_subset_names(self) -> list[str]:
        """Resolve the fixed subset set this benchmark instance must run."""

        if self.dataset_names is None:
            return list(MMEB_RUNTIME_REGISTRY)
        if not self.dataset_names:
            raise ValueError("MMEB datasets allowlist must not be empty.")
        unknown = [name for name in self.dataset_names if name not in MMEB_RUNTIME_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown MMEB subsets requested: {', '.join(unknown)}"
            )
        return list(self.dataset_names)

    def _resolve_datasets(self) -> None:
        """Resolve and validate the subset artifacts this benchmark must run.

        The benchmark subset set is determined only by the static benchmark
        index and
        the optional ``datasets`` allowlist. The filesystem is used only to
        validate artifact completeness, never to shrink the benchmark itself.
        Optional metadata self-checks are delegated to the dataset loader.
        """
        self._dataset_meta = {}
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"MMEB benchmark root not found: {self.data_path}")

        missing_subsets: list[str] = []
        missing_artifacts: list[str] = []
        expected_subset_names = self._resolve_expected_subset_names()

        for name in expected_subset_names:
            dataset_dir = os.path.join(self.data_path, PATH_REL_BY_DATASET[name])
            if not os.path.isdir(dataset_dir):
                missing_subsets.append(name)
                continue

            required_artifacts = ("queries.lance", "candidates.lance", "qrels.lance")
            missing_for_subset = [
                artifact
                for artifact in required_artifacts
                if not os.path.exists(os.path.join(dataset_dir, artifact))
            ]
            if missing_for_subset:
                missing_artifacts.append(f"{name}: {', '.join(missing_for_subset)}")
                continue

            self._dataset_meta[name] = {
                "modality": MODALITY_BY_DATASET[name],
                "task_type": TASK_TYPE_BY_DATASET[name],
                "path": dataset_dir,
            }

        if missing_subsets:
            raise FileNotFoundError(
                f"Missing MMEB subset directories under {self.data_path}: "
                f"{', '.join(missing_subsets)}"
            )
        if missing_artifacts:
            raise FileNotFoundError(
                "Incomplete MMEB subset artifacts: " + "; ".join(missing_artifacts)
            )

        logger.debug(
            f"Resolved {len(self._dataset_meta)} datasets from expected set "
            f"(total expected: {len(expected_subset_names)})"
        )
        for modality in ("image", "video", "visdoc"):
            count = sum(1 for meta in self._dataset_meta.values() if meta["modality"] == modality)
            if count > 0:
                logger.debug(f"  {modality}: {count} datasets")

    def aggregate(self, results: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Aggregate per-subset retrieval scores into benchmark summaries.

        Args:
            results: Raw benchmark results keyed by subset name. The benchmark
                ignores any internal keys beginning with ``"_"``.

        Returns:
            A summary dictionary containing:
            - per-task-type averages within each modality;
            - per-modality averages using the modality-specific primary metric;
            - an ``overall`` average across all discovered subsets.
        """
        grouped: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for dataset_name, scores in results.items():
            if dataset_name.startswith("_"):
                continue
            meta = self._dataset_meta.get(dataset_name)
            if meta is None:
                continue
            modality = cast(str, meta["modality"])
            task_type = cast(str, meta["task_type"])
            agg_metric = MODALITY_METRIC.get(modality, "hit@1")
            if agg_metric not in scores:
                raise ValueError(
                    f"{dataset_name}: missing aggregation metric {agg_metric!r} in scores"
                )
            agg_score = scores[agg_metric]
            grouped.setdefault(modality, {}).setdefault(task_type, []).append((dataset_name, agg_score))

        summary: dict[str, Any] = {}
        for modality in ("image", "video", "visdoc"):
            if modality not in grouped:
                continue
            all_scores: list[float] = []
            for task_type, dataset_scores in sorted(grouped[modality].items()):
                values = [score for _, score in dataset_scores]
                summary[task_type] = round(sum(values) / len(values), 4)
                all_scores.extend(values)
            summary[modality] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

        all_dataset_scores = []
        for modality_groups in grouped.values():
            for dataset_scores in modality_groups.values():
                all_dataset_scores.extend(score for _, score in dataset_scores)
        if all_dataset_scores:
            summary["overall"] = round(sum(all_dataset_scores) / len(all_dataset_scores), 4)
        return summary
