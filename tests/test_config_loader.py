from pathlib import Path
import textwrap

import pytest
from omegaconf import DictConfig, OmegaConf

from vlm2emb.config import load_config, to_native_config
from vlm2emb.exceptions import ConfigInheritanceError


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _assert_no_inherit(node: object) -> None:
    if isinstance(node, DictConfig):
        assert "_inherit_" not in node
        for value in node.values():
            _assert_no_inherit(value)
        return

    if isinstance(node, dict):
        assert "_inherit_" not in node
        for value in node.values():
            _assert_no_inherit(value)
        return

    if isinstance(node, list):
        for item in node:
            _assert_no_inherit(item)


def test_supports_linear_inheritance_chain(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "base.yaml",
        """
        defaults:
          precision: bf16
        train:
          args:
            epochs: 2
            batch_size: 16
        """,
    )
    _write_yaml(
        tmp_path / "mid.yaml",
        """
        _inherit_: ./base.yaml
        train:
          args:
            batch_size: 32
        model:
          name: qwen2vl
        """,
    )
    _write_yaml(
        tmp_path / "leaf.yaml",
        """
        _inherit_: ./mid.yaml
        train:
          args:
            learning_rate: 2e-5
        """,
    )

    config = load_config(tmp_path / "leaf.yaml")

    assert config.defaults.precision == "bf16"
    assert config.model.name == "qwen2vl"
    assert config.train.args.epochs == 2
    assert config.train.args.batch_size == 32
    assert config.train.args.learning_rate == 2e-5
    _assert_no_inherit(config)


def test_btoks_default_preset_uses_public_generation_protocol_and_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM2EMB_DATA_ROOT", raising=False)

    config = load_config("configs/presets/btoks_qwen2vl_2b_v1.yaml")

    assert config.train.args.generation_prefix_text == "<|im_start|>assistant\n"
    assert config.train.args.generation_suffix_text == "<|im_end|>"
    assert (
        config.train.dataset.datasets["visrag-indomain"].path
        == "./data/openbmb_VisRAG-Ret-Train-In-domain-data"
    )


def test_allows_shared_base_dag_reuse(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "shared.yaml",
        """
        defaults:
          source: shared
        train:
          args:
            batch_size: 8
        """,
    )
    _write_yaml(
        tmp_path / "left.yaml",
        """
        _inherit_: ./shared.yaml
        branch:
          side: left
        train:
          args:
            batch_size: 16
        """,
    )
    _write_yaml(
        tmp_path / "right.yaml",
        """
        _inherit_: ./shared.yaml
        branch:
          side: right
        """,
    )
    _write_yaml(
        tmp_path / "root.yaml",
        """
        graph:
          left:
            _inherit_: ./left.yaml
          right:
            _inherit_: ./right.yaml
        """,
    )

    config = load_config(tmp_path / "root.yaml")

    assert config.graph.left.defaults.source == "shared"
    assert config.graph.right.defaults.source == "shared"
    assert config.graph.left.branch.side == "left"
    assert config.graph.right.branch.side == "right"
    assert config.graph.left.train.args.batch_size == 16
    assert config.graph.right.train.args.batch_size == 8
    _assert_no_inherit(config)


def test_expands_root_then_nested_inherit(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "nested.yaml",
        """
        optimizer:
          type: adamw
        scheduler:
          name: cosine
        """,
    )
    _write_yaml(
        tmp_path / "base.yaml",
        """
        experiment:
          name: baseline
        train:
          args:
            batch_size: 32
          optimization:
            _inherit_: ./nested.yaml
            scheduler:
              warmup_steps: 100
        """,
    )
    _write_yaml(
        tmp_path / "root.yaml",
        """
        _inherit_: ./base.yaml
        train:
          args:
            batch_size: 64
        """,
    )

    config = load_config(tmp_path / "root.yaml")

    assert config.experiment.name == "baseline"
    assert config.train.args.batch_size == 64
    assert config.train.optimization.optimizer.type == "adamw"
    assert config.train.optimization.scheduler.name == "cosine"
    assert config.train.optimization.scheduler.warmup_steps == 100
    _assert_no_inherit(config)


def test_resolves_relative_inherit_paths(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "base.yaml",
        """
        data:
          root: ./datasets
        """,
    )
    _write_yaml(
        tmp_path / "configs" / "nested" / "child.yaml",
        """
        _inherit_: ../../base.yaml
        data:
          split: train
        """,
    )

    config = load_config(tmp_path / "configs" / "nested" / "child.yaml")

    assert config.data.root == "./datasets"
    assert config.data.split == "train"
    _assert_no_inherit(config)


def test_rejects_only_active_cycle_chain(tmp_path: Path) -> None:
    cycle_a = tmp_path / "cycle_a.yaml"
    cycle_b = tmp_path / "cycle_b.yaml"

    _write_yaml(
        cycle_a,
        """
        _inherit_: ./cycle_b.yaml
        model:
          name: cycle-a
        """,
    )
    _write_yaml(
        cycle_b,
        """
        _inherit_: ./cycle_a.yaml
        model:
          name: cycle-b
        """,
    )

    with pytest.raises(ConfigInheritanceError) as exc_info:
        load_config(cycle_a)

    chain = [Path(item).name for item in exc_info.value.inheritance_chain]
    assert chain == ["cycle_a.yaml", "cycle_b.yaml", "cycle_a.yaml"]
    assert exc_info.value.error_type == "circular"
    assert "cycle_a.yaml" in str(exc_info.value)
    assert "cycle_b.yaml" in str(exc_info.value)
    assert "shared.yaml" not in str(exc_info.value)
    assert "root.yaml" not in str(exc_info.value)


def test_to_native_config_returns_python_containers() -> None:
    config = OmegaConf.create(
        {
            "train": {
                "output_dir": "${experiment.name}",
            },
            "experiment": {
                "name": "demo-run",
            },
        }
    )

    native = to_native_config(config)

    assert native == {
        "train": {"output_dir": "demo-run"},
        "experiment": {"name": "demo-run"},
    }
    assert to_native_config(native) is native
