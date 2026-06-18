from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

import vlm2emb.model as model_module
from vlm2emb.model import VLM2Emb, VLM2EmbConfig


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


def _write_checkpoint(tmp_path, modules: dict) -> None:
    config = VLM2EmbConfig(modules=modules)
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()) + "\n")
    save_file({}, tmp_path / "model.safetensors")


def test_reference_mode_load_hydrates_skeleton_before_rematerialize(tmp_path, monkeypatch):
    _write_checkpoint(
        tmp_path,
        {
            "backbone": {
                "type": "DummyBackbone",
                "model_name_or_path": "dummy/backbone",
                "trust_remote_code": True,
            }
        },
    )

    calls: list[dict] = []

    def fake_module_from_config(config):
        calls.append(dict(config))
        return DummyModule()

    def fake_config_from_pretrained(path, **kwargs):
        assert path == "dummy/backbone"
        assert kwargs["trust_remote_code"] is True
        return SimpleNamespace(to_dict=lambda: {"model_type": "dummy", "hidden_size": 4})

    monkeypatch.setattr(model_module.AutoModule, "from_config", fake_module_from_config)
    monkeypatch.setattr(model_module.HFAutoConfig, "from_pretrained", fake_config_from_pretrained)

    model = VLM2Emb.from_pretrained(tmp_path)

    backbone_calls = [call for call in calls if call["type"] == "DummyBackbone"]
    assert len(backbone_calls) == 2
    assert "backbone_config" in backbone_calls[0]
    assert backbone_calls[0]["model_name_or_path"] == "dummy/backbone"
    assert "backbone_config" in model.config.modules["backbone"]
    assert model.config.modules["backbone"]["model_name_or_path"] == "dummy/backbone"


def test_full_mode_load_does_not_rematerialize_reference_modules(tmp_path, monkeypatch):
    _write_checkpoint(
        tmp_path,
        {
            "backbone": {
                "type": "DummyBackbone",
                "backbone_config": {"model_type": "dummy", "hidden_size": 4},
            }
        },
    )

    calls: list[dict] = []

    def fake_module_from_config(config):
        calls.append(dict(config))
        return DummyModule()

    def fail_config_load(*args, **kwargs):
        raise AssertionError("full mode should not hydrate reference module configs")

    monkeypatch.setattr(model_module.AutoModule, "from_config", fake_module_from_config)
    monkeypatch.setattr(model_module.HFAutoConfig, "from_pretrained", fail_config_load)

    VLM2Emb.from_pretrained(tmp_path)

    backbone_calls = [call for call in calls if call["type"] == "DummyBackbone"]
    assert len(backbone_calls) == 1
    assert "backbone_config" in backbone_calls[0]
    assert "model_name_or_path" not in backbone_calls[0]
