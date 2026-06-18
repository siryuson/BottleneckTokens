"""Tests for standalone evaluation loading and fallback behavior."""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest


class DummyAccelerator:
    """Minimal accelerator stub for eval tests."""

    def __init__(self):
        self.device = "cpu"
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0
        self.wait_called = False
        self.end_training_called = False

    def autocast(self):
        return nullcontext()

    def wait_for_everyone(self):
        self.wait_called = True

    def end_training(self):
        self.end_training_called = True


class DummyNonMainAccelerator(DummyAccelerator):
    """Non-main accelerator stub for distributed eval tests."""

    def __init__(self):
        super().__init__()
        self.is_main_process = False
        self.num_processes = 2
        self.process_index = 1


class DummyModel:
    """Minimal model stub for eval tests."""

    def __init__(self):
        self.setup_processor = None
        self.moved_to = None
        self.eval_called = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True

    def setup(self, processor):
        self.setup_processor = processor


class DummyBenchmark:
    """Simple benchmark stub."""

    name = "dummy"

    def __len__(self):
        return 1

    def __call__(self, model, processor_wrapper, accelerator):
        return {
            "dataset-a": {"hit@1": 0.5},
            "_summary": {"overall": 0.5},
        }


class TestProcessorInference:
    """Test processor inference helper behavior."""

    def test_processor_fallback_to_second_path(self, tmp_path):
        """Primary path fails and fallback path is included in the error."""
        from vlm2emb.evaluation.evaluate import _load_processor

        primary = tmp_path / "primary"
        primary.mkdir()
        fallback = tmp_path / "fallback"
        fallback.mkdir()

        with pytest.raises(ValueError, match="or"):
            _load_processor(str(primary), fallback=str(fallback))

    def test_processor_no_fallback_raises(self, tmp_path):
        """No fallback path still raises a clear error."""
        from vlm2emb.evaluation.evaluate import _load_processor

        with pytest.raises(ValueError, match="Cannot load processor"):
            _load_processor(str(tmp_path / "nonexistent"))


class TestEvaluateConfigValidation:
    """Test high-level config validation."""

    def test_missing_eval_key(self):
        from vlm2emb.evaluation.evaluate import evaluate

        with pytest.raises(ValueError, match="eval"):
            evaluate({"model": {"type": "VLM2Emb"}})

    def test_missing_checkpoint_and_model(self):
        from vlm2emb.evaluation.evaluate import evaluate

        with pytest.raises(ValueError, match="runtime checkpoint argument or a 'model' section"):
            evaluate({"eval": {"type": "mmeb"}})


class TestStandaloneFallbacks:
    """Test standalone checkpoint loading fallbacks."""

    def test_full_checkpoint_uses_config_processor_fallback(self, monkeypatch, caplog):
        from vlm2emb.evaluation import evaluate as eval_mod

        dummy_model = DummyModel()
        dummy_processor = object()

        monkeypatch.setattr(eval_mod, "Accelerator", DummyAccelerator)
        monkeypatch.setattr(eval_mod, "_try_load_peft_config", lambda _: None)
        monkeypatch.setattr(
            eval_mod.VLM2Emb,
            "from_pretrained",
            classmethod(lambda cls, path, *args, **kwargs: dummy_model),
        )
        monkeypatch.setattr(
            eval_mod,
            "_load_processor",
            lambda path, fallback=None: (_ for _ in ()).throw(ValueError("missing")),
        )
        monkeypatch.setattr(eval_mod, "_load_processor_from_config", lambda config: dummy_processor)

        with caplog.at_level("WARNING"):
            model, processor, _ = eval_mod.load_model_and_processor(
                {"eval": {"type": "mmeb"}, "processor": {"type": "dummy"}},
                checkpoint="repo/full-checkpoint",
            )

        assert model is dummy_model
        assert processor is dummy_processor
        assert dummy_model.setup_processor is dummy_processor
        assert "falling back to config initialization" in caplog.text

    def test_peft_checkpoint_falls_back_to_config_with_warning(self, monkeypatch, caplog):
        from vlm2emb.evaluation import evaluate as eval_mod

        dummy_model = DummyModel()
        dummy_processor = object()

        peft_module = types.ModuleType("peft")

        class _AutoPeftModel:
            @staticmethod
            def from_pretrained(path):
                raise RuntimeError("direct peft load failed")

        class _PeftModel:
            @staticmethod
            def from_pretrained(model, path):
                class _Merged:
                    @staticmethod
                    def merge_and_unload():
                        return model

                return _Merged()

        peft_module.AutoPeftModel = _AutoPeftModel
        peft_module.PeftModel = _PeftModel
        monkeypatch.setitem(sys.modules, "peft", peft_module)

        monkeypatch.setattr(eval_mod, "Accelerator", DummyAccelerator)
        monkeypatch.setattr(
            eval_mod,
            "_try_load_peft_config",
            lambda _: types.SimpleNamespace(base_model_name_or_path="/missing/base"),
        )
        monkeypatch.setattr(eval_mod, "create_model_and_processor", lambda config: (dummy_model, dummy_processor))
        monkeypatch.setattr(
            eval_mod,
            "_load_processor",
            lambda path, fallback=None: (_ for _ in ()).throw(ValueError("missing")),
        )

        with caplog.at_level("WARNING"):
            model, processor, _ = eval_mod.load_model_and_processor(
                {
                    "eval": {"type": "mmeb"},
                    "model": {"type": "vlm2emb"},
                    "processor": {"type": "dummy"},
                },
                checkpoint="repo/adapter",
            )

        assert model is dummy_model
        assert processor is dummy_processor
        assert "Direct PEFT checkpoint load failed" in caplog.text
        assert "using processor initialized from config" in caplog.text

    def test_peft_without_model_fallback_raises(self, monkeypatch):
        from vlm2emb.evaluation import evaluate as eval_mod

        peft_module = types.ModuleType("peft")

        class _AutoPeftModel:
            @staticmethod
            def from_pretrained(path):
                raise RuntimeError("direct peft load failed")

        peft_module.AutoPeftModel = _AutoPeftModel
        monkeypatch.setitem(sys.modules, "peft", peft_module)
        monkeypatch.setattr(eval_mod, "Accelerator", DummyAccelerator)
        monkeypatch.setattr(
            eval_mod,
            "_try_load_peft_config",
            lambda _: types.SimpleNamespace(base_model_name_or_path="/missing/base"),
        )

        with pytest.raises(ValueError, match="no model fallback"):
            eval_mod.load_model_and_processor(
                {"eval": {"type": "mmeb"}},
                checkpoint="repo/adapter",
            )


class TestEvaluateEntrypoint:
    """Test benchmark execution wiring."""

    def test_evaluate_runs_benchmark_with_runtime_checkpoint(self, monkeypatch, tmp_path):
        from vlm2emb.evaluation import evaluate as eval_mod

        dummy_model = DummyModel()
        dummy_processor = object()
        saved: dict[str, object] = {}

        class InspectBenchmark(DummyBenchmark):
            def __call__(self, model, processor_wrapper, accelerator):
                assert accelerator.process_index == 0
                assert accelerator.num_processes == 1
                assert accelerator.is_main_process is True
                return super().__call__(model, processor_wrapper, accelerator)

        monkeypatch.setattr(
            eval_mod,
            "load_model_and_processor",
            lambda config, checkpoint=None: (dummy_model, dummy_processor, DummyAccelerator()),
        )
        monkeypatch.setattr(eval_mod, "create_processor_wrapper", lambda processor, **kwargs: "wrapper")
        monkeypatch.setattr(eval_mod.AutoBenchmark, "from_config", lambda eval_config: InspectBenchmark())
        monkeypatch.setattr(eval_mod, "_resolve_output_dir", lambda checkpoint, output_dir: tmp_path)
        monkeypatch.setattr(
            eval_mod,
            "save_results_payload",
            lambda path, results, meta: saved.update({"results_path": path, "results": results, "meta": meta}),
        )
        monkeypatch.setattr(
            eval_mod,
            "save_resolved_config",
            lambda path, config: saved.update({"config_path": path, "config": config}),
        )

        results = eval_mod.evaluate(
            {"eval": {"type": "mmeb"}},
            checkpoint="repo/runtime",
        )

        assert results["dataset-a"]["hit@1"] == 0.5
        assert results["_summary"]["overall"] == 0.5
        assert saved["meta"] == {
            "checkpoint": "repo/runtime",
            "config_path": None,
            "world_size": 1,
        }
        assert str(saved["results_path"]).endswith("results.json")
        assert str(saved["config_path"]).endswith("config.yaml")

    def test_evaluate_passes_wrapper_only_processor_kwargs(self, monkeypatch, tmp_path):
        from vlm2emb.evaluation import evaluate as eval_mod

        dummy_model = DummyModel()
        dummy_processor = object()
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            eval_mod,
            "load_model_and_processor",
            lambda config, checkpoint=None: (dummy_model, dummy_processor, DummyAccelerator()),
        )
        monkeypatch.setattr(
            eval_mod,
            "create_processor_wrapper",
            lambda processor, **kwargs: captured.update({"processor": processor, "kwargs": kwargs}) or "wrapper",
        )
        monkeypatch.setattr(eval_mod.AutoBenchmark, "from_config", lambda eval_config: DummyBenchmark())
        monkeypatch.setattr(eval_mod, "_resolve_output_dir", lambda checkpoint, output_dir: tmp_path)
        monkeypatch.setattr(eval_mod, "save_results_payload", lambda *args, **kwargs: None)
        monkeypatch.setattr(eval_mod, "save_resolved_config", lambda *args, **kwargs: None)

        eval_mod.evaluate(
            {
                "eval": {"type": "mmeb"},
                "processor": {
                    "type": "qwen2_vl",
                    "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
                    "image_min_pixels": 3136,
                    "image_max_pixels": 1003520,
                    "wrap_visual_tokens_with_boundaries": False,
                },
            },
            checkpoint="repo/runtime",
        )

        assert captured["processor"] is dummy_processor
        assert captured["kwargs"] == {"wrap_visual_tokens_with_boundaries": False}

    def test_evaluate_rejects_incomplete_accelerator_contract(self, monkeypatch):
        from vlm2emb.evaluation import evaluate as eval_mod

        dummy_model = DummyModel()
        dummy_processor = object()

        incomplete_accelerator = types.SimpleNamespace(
            device="cpu",
            is_main_process=True,
            num_processes=1,
            wait_for_everyone=lambda: None,
            end_training=lambda: None,
            autocast=lambda: nullcontext(),
        )

        monkeypatch.setattr(
            eval_mod,
            "load_model_and_processor",
            lambda config, checkpoint=None: (
                dummy_model,
                dummy_processor,
                incomplete_accelerator,
            ),
        )

        with pytest.raises(TypeError, match="process_index"):
            eval_mod.evaluate(
                {"eval": {"type": "mmeb"}},
                checkpoint="repo/runtime",
            )

    def test_evaluate_non_main_uses_consistent_results_without_saving(self, monkeypatch, tmp_path):
        from vlm2emb.evaluation import evaluate as eval_mod

        dummy_model = DummyModel()
        dummy_processor = object()
        accelerator = DummyNonMainAccelerator()
        info_messages: list[str] = []

        monkeypatch.setattr(
            eval_mod,
            "load_model_and_processor",
            lambda config, checkpoint=None: (dummy_model, dummy_processor, accelerator),
        )
        monkeypatch.setattr(eval_mod, "create_processor_wrapper", lambda processor, **kwargs: "wrapper")
        monkeypatch.setattr(eval_mod.AutoBenchmark, "from_config", lambda eval_config: DummyBenchmark())
        monkeypatch.setattr(eval_mod, "_resolve_output_dir", lambda checkpoint, output_dir: tmp_path)
        monkeypatch.setattr(
            eval_mod,
            "save_results_payload",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("non-main process must not save results")
            ),
        )
        monkeypatch.setattr(
            eval_mod,
            "save_resolved_config",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("non-main process must not save config")
            ),
        )
        monkeypatch.setattr(
            eval_mod.logger,
            "info",
            lambda message, *args, **kwargs: info_messages.append(str(message)),
        )

        results = eval_mod.evaluate(
            {"eval": {"type": "mmeb"}},
            checkpoint="repo/runtime",
        )

        assert results["dataset-a"]["hit@1"] == 0.5
        assert results["_summary"] == {"overall": 0.5}
        assert accelerator.wait_called is True
        assert accelerator.end_training_called is True
        assert "Evaluation summary:" not in info_messages
