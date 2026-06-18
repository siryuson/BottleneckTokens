from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vlm2emb.evaluation.benchmarks.base import BaseBenchmark
from vlm2emb.training import trainer as trainer_mod


class _Benchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="dummy")
        self.calls: list[dict[str, Any]] = []

    @property
    def evaluators(self):
        return []

    def aggregate(self, results):
        return {"overall": 0.8}

    def __call__(self, model, processor_wrapper, accelerator=None, **kwargs):
        self.calls.append(
            {
                "model": model,
                "processor_wrapper": processor_wrapper,
                "accelerator": accelerator,
            }
        )
        return {"dataset-a": {"hit@1": 0.8}, "_summary": {"overall": 0.8}}


class _CallbackHandler:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def on_evaluate(self, args, state, control, metrics):
        self.calls.append({"args": args, "state": state, "control": control, "metrics": dict(metrics)})
        control.should_evaluate = False
        return control


class _MemoryTracker:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop_and_update_metrics(self, metrics):
        self.stopped += 1
        metrics["eval_mem_cpu_alloc_delta"] = 1.0


def _make_trainer(*, is_main_process: bool):
    trainer = trainer_mod.Trainer.__new__(trainer_mod.Trainer)
    trainer.eval_dataset = _Benchmark()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.processing_class = object()
    trainer.accelerator = SimpleNamespace(
        is_main_process=is_main_process,
        num_processes=1,
        device=torch.device("cpu"),
    )
    trainer.args = SimpleNamespace()
    trainer.state = SimpleNamespace()
    trainer.control = SimpleNamespace(should_evaluate=True)
    trainer.callback_handler = _CallbackHandler()
    trainer._memory_tracker = _MemoryTracker()
    trainer.logged_metrics: list[dict[str, float]] = []
    trainer.saved_results: list[dict[str, Any]] = []
    trainer.log = lambda metrics: trainer.logged_metrics.append(dict(metrics))
    trainer._distributed_max_float = lambda elapsed: 9.876
    trainer._save_eval_results = lambda results: trainer.saved_results.append(dict(results))
    return trainer


def test_benchmark_evaluate_runs_log_and_callback_on_non_main_rank():
    trainer = _make_trainer(is_main_process=False)

    metrics = trainer_mod.Trainer.evaluate(trainer)

    logged_metrics = {"eval_overall": 0.8, "eval_runtime": 9.88}
    assert metrics == {**logged_metrics, "eval_mem_cpu_alloc_delta": 1.0}
    assert trainer.logged_metrics == [logged_metrics]
    assert trainer.callback_handler.calls[0]["metrics"] == logged_metrics
    assert trainer.control.should_evaluate is False
    assert trainer._memory_tracker.started == 1
    assert trainer._memory_tracker.stopped == 1
    assert trainer.saved_results == []


def test_benchmark_evaluate_saves_results_only_on_main_rank():
    trainer = _make_trainer(is_main_process=True)

    metrics = trainer_mod.Trainer.evaluate(trainer)

    logged_metrics = {"eval_overall": 0.8, "eval_runtime": 9.88}
    assert metrics == {**logged_metrics, "eval_mem_cpu_alloc_delta": 1.0}
    assert trainer.logged_metrics == [logged_metrics]
    assert trainer.callback_handler.calls[0]["metrics"] == logged_metrics
    assert trainer.control.should_evaluate is False
    assert trainer._memory_tracker.started == 1
    assert trainer._memory_tracker.stopped == 1
    assert trainer.saved_results == [
        {"dataset-a": {"hit@1": 0.8}, "_summary": {"overall": 0.8}}
    ]


def test_distributed_max_float_uses_accelerator_gather_max():
    trainer = trainer_mod.Trainer.__new__(trainer_mod.Trainer)
    captured: dict[str, Any] = {}

    def fake_gather(tensor):
        captured["local"] = float(tensor.item())
        return torch.tensor([3.0, 12.5], dtype=tensor.dtype)

    trainer.accelerator = SimpleNamespace(
        num_processes=2,
        device=torch.device("cpu"),
        gather=fake_gather,
    )

    assert trainer_mod.Trainer._distributed_max_float(trainer, 3.0) == 12.5
    assert captured["local"] == 3.0


def test_save_eval_results_uses_eval_only_directory(tmp_path):
    trainer = trainer_mod.Trainer.__new__(trainer_mod.Trainer)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.state = SimpleNamespace(global_step=500)

    trainer_mod.Trainer._save_eval_results(
        trainer,
        {"dataset-a": {"hit@1": 0.8}, "_summary": {"overall": 0.8}},
    )

    assert (tmp_path / "eval-results" / "step-500" / "eval_results.json").exists()
    assert not (tmp_path / "checkpoint-500").exists()
