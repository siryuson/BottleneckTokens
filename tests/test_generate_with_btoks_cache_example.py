from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path("examples/model/generate_with_btoks_cache.py")
SPEC = importlib.util.spec_from_file_location("generate_with_btoks_cache_example", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_generation_protocol_uses_training_config_defaults():
    prompt, suffix = MODULE._resolve_generation_protocol(
        {
            "train": {
                "args": {
                    "generation_prefix_text": "<|im_start|>assistant\n",
                    "generation_suffix_text": "<|im_end|>",
                }
            }
        },
        generation_prompt_override=None,
        generation_suffix_override=None,
    )

    assert prompt == "<|im_start|>assistant\n"
    assert suffix == "<|im_end|>"


def test_resolve_generation_protocol_allows_cli_overrides():
    prompt, suffix = MODULE._resolve_generation_protocol(
        {
            "train": {
                "args": {
                    "generation_prefix_text": "cfg-prompt",
                    "generation_suffix_text": "cfg-suffix",
                }
            }
        },
        generation_prompt_override="cli-prompt",
        generation_suffix_override="cli-suffix",
    )

    assert prompt == "cli-prompt"
    assert suffix == "cli-suffix"


def test_set_attn_implementation_updates_backbone_modules():
    config = {
        "model": {
            "modules": {
                "injector": {"type": "BToksTokenInjector"},
                "backbone": {
                    "type": "Qwen2VLBackbone",
                    "attn_implementation": "flash_attention_2",
                },
            }
        }
    }

    MODULE._set_attn_implementation(config, "eager")

    assert config["model"]["modules"]["backbone"]["attn_implementation"] == "eager"


def test_btoks_model_type_counts_as_saved_artifact(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "btoks"}),
        encoding="utf-8",
    )

    assert MODULE._is_vlm2emb_artifact(str(tmp_path))


def test_vlm2emb_model_type_still_counts_as_saved_artifact(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "vlm2emb"}),
        encoding="utf-8",
    )

    assert MODULE._is_vlm2emb_artifact(str(tmp_path))
