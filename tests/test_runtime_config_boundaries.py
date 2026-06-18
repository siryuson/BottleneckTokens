import argparse
import importlib
from types import SimpleNamespace
from typing import Any

from omegaconf import DictConfig, OmegaConf

import vlm2emb
import vlm2emb.config as config_mod
import vlm2emb.data.processors as processor_mod
import vlm2emb.evaluation.evaluate as evaluate_mod
import vlm2emb.model as model_mod
import vlm2emb.training.train as train_mod
from vlm2emb.registry import Registry


class DummyPartialState:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.process_index = 0
        self.num_processes = 1
        self.is_main_process = True

    def wait_for_everyone(self) -> None:
        return None


class DummyAccelerator:
    def wait_for_everyone(self) -> None:
        return None

    def end_training(self) -> None:
        return None


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace()

    def parameters(self) -> list[Any]:
        return []


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )


class FakeFactory:
    def __init__(self, builder, contains: set[str] | None = None) -> None:
        self.builder = builder
        self.contains = contains or set()
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __contains__(self, name: str) -> bool:
        return name in self.contains

    def from_config(self, config: Any, **kwargs: Any) -> Any:
        self.calls.append((config, kwargs))
        return self.builder(config, **kwargs)


class DummyTrainer:
    def __init__(self) -> None:
        self.accelerator = DummyAccelerator()
        self.resume_from_checkpoint: str | None = None

    def train(self, resume_from_checkpoint: str | None = None) -> None:
        self.resume_from_checkpoint = resume_from_checkpoint


def test_train_script_normalizes_config_before_invoking_train(monkeypatch):
    train_script = importlib.import_module("scripts.train")
    config = OmegaConf.create(
        {
            "model": {"type": "mock-model"},
            "train": {
                "args": {"output_dir": "./output/train"},
                "dataset": {"type": "mock-dataset"},
                "collator": {"type": "mock-collator"},
                "trainer": {"type": "mock-trainer"},
            },
        }
    )
    captured: dict[str, Any] = {}

    class DummyLoader:
        def load_with_inheritance(self, path: str, resolve_interpolation: bool = True):
            assert path == "configs/mock-train.yaml"
            assert resolve_interpolation is False
            return config

    monkeypatch.setattr(train_script, "parse_args", lambda: argparse.Namespace(
        config="configs/mock-train.yaml",
        overrides=[],
        verbose=False,
        log_file=None,
    ))
    monkeypatch.setattr(train_script, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script, "set_verbosity", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.logger, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(config_mod, "ConfigLoader", lambda: DummyLoader())
    monkeypatch.setattr(config_mod, "apply_overrides", lambda cfg, overrides: cfg)
    monkeypatch.setattr(vlm2emb, "train", lambda cfg: captured.setdefault("config", cfg))

    train_script.main()

    assert isinstance(captured["config"], dict)
    assert not isinstance(captured["config"], DictConfig)
    assert captured["config"]["train"]["args"]["output_dir"] == "./output/train"


def test_train_btoks_module_shim_runs_without_repo_relative_script(monkeypatch):
    train_btoks = importlib.import_module("vlm2emb.cli.train_btoks")
    config = OmegaConf.create(
        {
            "model": {"type": "mock-model"},
            "train": {
                "args": {"output_dir": "./output/train"},
                "dataset": {"type": "mock-dataset"},
                "collator": {"type": "mock-collator"},
                "trainer": {"type": "mock-trainer"},
            },
        }
    )
    captured: dict[str, Any] = {}

    class DummyLoader:
        def load_with_inheritance(self, path: str, resolve_interpolation: bool = True):
            assert path == "configs/mock-train.yaml"
            assert resolve_interpolation is False
            return config

    monkeypatch.setattr(train_btoks, "parse_args", lambda: argparse.Namespace(
        config="configs/mock-train.yaml",
        overrides=[],
        verbose=False,
        log_file=None,
    ))
    monkeypatch.setattr(train_btoks, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_btoks, "set_verbosity", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_btoks, "ConfigLoader", lambda: DummyLoader())
    monkeypatch.setattr(train_btoks, "apply_overrides", lambda cfg, overrides: cfg)
    monkeypatch.setattr(train_btoks, "train", lambda cfg: captured.setdefault("config", cfg))

    assert train_btoks.main() == 0
    assert isinstance(captured["config"], dict)
    assert captured["config"]["train"]["args"]["output_dir"] == "./output/train"


def test_eval_script_normalizes_config_before_invoking_evaluate(monkeypatch):
    eval_script = importlib.import_module("scripts.eval")
    config = OmegaConf.create(
        {
            "eval": {"type": "mock-benchmark", "name": "demo"},
        }
    )
    captured: dict[str, Any] = {}

    class DummyLoader:
        def load_with_inheritance(self, path: str, resolve_interpolation: bool = True):
            assert path == "configs/mock-eval.yaml"
            assert resolve_interpolation is False
            return config

    monkeypatch.setattr(eval_script, "parse_args", lambda: argparse.Namespace(
        config="configs/mock-eval.yaml",
        overrides=[],
        verbose=False,
        log_file=None,
        checkpoint="/tmp/mock-checkpoint",
        output_dir="/tmp/mock-output",
    ))
    monkeypatch.setattr(eval_script, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_script, "set_verbosity", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_script.logger, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(config_mod, "ConfigLoader", lambda: DummyLoader())
    monkeypatch.setattr(config_mod, "apply_overrides", lambda cfg, overrides: cfg)

    def fake_evaluate(cfg, checkpoint=None, output_dir=None, config_path=None):
        captured["config"] = cfg
        captured["checkpoint"] = checkpoint
        captured["output_dir"] = output_dir
        captured["config_path"] = config_path
        return {"ok": True}

    monkeypatch.setattr(evaluate_mod, "evaluate", fake_evaluate)

    eval_script.main()

    assert isinstance(captured["config"], dict)
    assert not isinstance(captured["config"], DictConfig)
    assert captured["config"]["eval"]["type"] == "mock-benchmark"
    assert captured["checkpoint"] == "/tmp/mock-checkpoint"
    assert captured["output_dir"] == "/tmp/mock-output"
    assert captured["config_path"] == "configs/mock-eval.yaml"


def test_registry_from_config_converts_dictconfig_before_component_build():
    registry = Registry("runtime-boundary")
    captured: dict[str, Any] = {}

    class FromConfigComponent:
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> dict[str, Any]:
            captured["from_config"] = config
            return config

    class ConstructorComponent:
        def __init__(self, **kwargs: Any) -> None:
            captured["constructor"] = kwargs

    registry.register("from-config", FromConfigComponent)
    registry.register("constructor", ConstructorComponent)

    from_config_result = registry.from_config(
        OmegaConf.create({"type": "from-config", "nested": {"value": 1}})
    )
    registry.from_config(
        OmegaConf.create({"type": "constructor", "nested": {"value": 2}})
    )

    assert isinstance(captured["from_config"], dict)
    assert not isinstance(captured["from_config"], DictConfig)
    assert isinstance(captured["from_config"]["nested"], dict)
    assert from_config_result["nested"]["value"] == 1

    assert isinstance(captured["constructor"], dict)
    assert isinstance(captured["constructor"]["nested"], dict)
    assert not isinstance(captured["constructor"]["nested"], DictConfig)
    assert captured["constructor"]["nested"]["value"] == 2


def test_train_accepts_plain_dict_config(monkeypatch):
    DummyPartialState.calls = []
    args_factory = FakeFactory(
        lambda config, **kwargs: SimpleNamespace(
            output_dir=config["output_dir"],
            report_to=["none"],
            resume_from_checkpoint=None,
            ddp_timeout=config.get("ddp_timeout"),
        )
    )
    dataset_factory = FakeFactory(lambda config, **kwargs: {"dataset": config})
    benchmark_factory = FakeFactory(
        lambda config, **kwargs: {"benchmark": config},
        contains={"mock-benchmark"},
    )
    collator_factory = FakeFactory(
        lambda config, **kwargs: {
            "collator": config,
            "wrapper": kwargs["wrapper"],
        }
    )
    trainer_factory = FakeFactory(
        lambda config, **kwargs: DummyTrainer()
    )

    monkeypatch.setattr(train_mod, "PartialState", DummyPartialState)
    monkeypatch.setattr(train_mod, "AutoTrainingArgs", args_factory)
    monkeypatch.setattr(train_mod, "AutoDataset", dataset_factory)
    monkeypatch.setattr(train_mod, "AutoBenchmark", benchmark_factory)
    monkeypatch.setattr(train_mod, "AutoCollator", collator_factory)
    monkeypatch.setattr(train_mod, "AutoTrainer", trainer_factory)
    monkeypatch.setattr(
        model_mod,
        "create_model_and_processor",
        lambda config: (DummyModel(), DummyProcessor()),
    )
    monkeypatch.setattr(
        processor_mod,
        "create_processor_wrapper",
        lambda processor: {"processor": processor},
    )

    config = {
        "model": {"type": "mock-model"},
        "train": {
            "args": {"output_dir": "./output/train", "ddp_timeout": 1800},
            "dataset": {"type": "mock-dataset", "name": "train-set"},
            "collator": {"type": "mock-collator", "packing": True},
            "trainer": {"type": "mock-trainer", "training_args": "ignored"},
        },
        "eval": {"type": "mock-benchmark", "name": "eval-set"},
    }

    train_mod.train(config)

    assert DummyPartialState.calls[0]["timeout"].total_seconds() == 1800
    assert args_factory.calls[0][0] == config["train"]["args"]
    assert dataset_factory.calls[0][0] == config["train"]["dataset"]
    assert benchmark_factory.calls[0][0] == config["eval"]
    assert collator_factory.calls[0][0] == config["train"]["collator"]
    assert trainer_factory.calls[0][0] == {"type": "mock-trainer"}
    assert trainer_factory.calls[0][1]["train_dataset"] == {"dataset": config["train"]["dataset"]}
    assert trainer_factory.calls[0][1]["eval_dataset"] == {"benchmark": config["eval"]}


def test_btoks_train_entry_builds_cpu_dry_run_without_real_model_or_data(monkeypatch, tmp_path):
    DummyPartialState.calls = []
    dataset_factory = FakeFactory(lambda config, **kwargs: {"dataset": config})
    collator_factory = FakeFactory(
        lambda config, **kwargs: {
            "collator": config,
            "wrapper": kwargs["wrapper"],
        }
    )
    trainer_factory = FakeFactory(lambda config, **kwargs: DummyTrainer())

    monkeypatch.setattr(train_mod, "PartialState", DummyPartialState)
    monkeypatch.setattr(train_mod, "AutoDataset", dataset_factory)
    monkeypatch.setattr(train_mod, "AutoCollator", collator_factory)
    monkeypatch.setattr(train_mod, "AutoTrainer", trainer_factory)
    monkeypatch.setattr(
        model_mod,
        "create_model_and_processor",
        lambda config: (DummyModel(), DummyProcessor()),
    )
    monkeypatch.setattr(
        processor_mod,
        "create_processor_wrapper",
        lambda processor: {"processor": processor},
    )

    config = {
        "model": {"type": "BToks", "modules": {}},
        "train": {
            "args": {
                "type": "btoks",
                "output_dir": str(tmp_path / "btoks-dry-run"),
                "report_to": "none",
                "bf16": False,
                "use_cpu": True,
                "max_steps": 0,
            },
            "dataset": {"type": "mock-dataset", "name": "train-set"},
            "collator": {"type": "mock-collator"},
            "trainer": {"type": "btoks_trainer"},
        },
    }

    train_mod.train(config)

    training_args = trainer_factory.calls[0][1]["args"]
    assert type(training_args).__name__ == "BToksTrainingArgs"
    assert training_args.bf16 is False
    assert training_args.use_cpu is True
    assert training_args.max_steps == 0
    assert trainer_factory.calls[0][0] == {"type": "btoks_trainer"}
