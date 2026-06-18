from __future__ import annotations

from typing import Any

from vlm2emb.evaluation.benchmarks.base import BaseBenchmark


class _Runtime:
    def __init__(self, *, is_main_process: bool):
        self.is_main_process = is_main_process


class _Evaluator:
    name = "VOC2007"

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def __call__(
        self,
        model: Any,
        processor_wrapper: Any,
        accelerator: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return self.scores


class _Benchmark(BaseBenchmark):
    def __init__(self, scores: dict[str, float]):
        super().__init__(name="dummy")
        self._evaluators = [_Evaluator(scores)]
        self.aggregate_called = False

    @property
    def evaluators(self):
        return self._evaluators

    def aggregate(self, results: dict[str, dict[str, float]]) -> dict[str, Any]:
        self.aggregate_called = True
        if "hit@1" not in results["VOC2007"]:
            raise ValueError("missing hit@1")
        return {"overall": results["VOC2007"]["hit@1"]}


def test_base_benchmark_aggregates_on_main_process():
    benchmark = _Benchmark({"hit@1": 0.75})

    results = benchmark(
        model=object(),
        processor_wrapper=object(),
        accelerator=_Runtime(is_main_process=True),
    )

    assert benchmark.aggregate_called is True
    assert results["_summary"] == {"overall": 0.75}


def test_base_benchmark_aggregates_on_non_main_process_with_consistent_scores():
    benchmark = _Benchmark({"hit@1": 0.75})

    results = benchmark(
        model=object(),
        processor_wrapper=object(),
        accelerator=_Runtime(is_main_process=False),
    )

    assert benchmark.aggregate_called is True
    assert results == {"VOC2007": {"hit@1": 0.75}, "_summary": {"overall": 0.75}}
