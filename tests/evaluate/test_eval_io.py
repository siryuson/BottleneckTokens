"""Tests for evaluation result persistence helpers."""

from __future__ import annotations

import json

from omegaconf import OmegaConf


def test_save_results_payload_normalizes_summary_and_datasets(tmp_path):
    from vlm2emb.evaluation.io import save_results_payload

    save_path = tmp_path / "results.json"
    save_results_payload(
        save_path,
        {
            "dataset-a": {"hit@1": 0.5},
            "_summary": {"overall": 0.5},
        },
        meta={"checkpoint": "repo/model"},
    )

    payload = json.loads(save_path.read_text(encoding="utf-8"))
    assert payload["meta"]["checkpoint"] == "repo/model"
    assert payload["summary"]["overall"] == 0.5
    assert payload["datasets"]["dataset-a"]["hit@1"] == 0.5
    assert "_summary" not in payload["datasets"]


def test_save_resolved_config_writes_yaml(tmp_path):
    from vlm2emb.evaluation.io import save_resolved_config

    config = OmegaConf.create(
        {
            "root": "./outputs",
            "eval": {"output_dir": "${root}/eval"},
        }
    )
    save_path = tmp_path / "config.yaml"
    save_resolved_config(save_path, config)

    text = save_path.read_text(encoding="utf-8")
    assert "output_dir: ./outputs/eval" in text
